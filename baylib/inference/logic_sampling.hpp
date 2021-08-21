//
// Created by elle on 22/07/21.
//

#ifndef BAYLIB_LOGIC_SAMPLING_HPP
#define BAYLIB_LOGIC_SAMPLING_HPP

#define DEBUG_MONTECARLO 0
#define MEMORY_SLACK 0.8
#define PRINT_N 100

#ifndef BOOST_COMPUTE_THREAD_SAFE
#define BOOST_COMPUTE_THREAD_SAFE
#endif

#include <vector>
#include <future>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

#include <baylib/network/bayesian_network.hpp>
#include <baylib/network/bayesian_utils.hpp>
#include <baylib/tools/threads/thread_pool.hpp>

namespace bn {
    namespace compute = boost::compute;
    using boost::compute::lambda::_1;
    using boost::compute::lambda::_2;

    struct bcvec {
        compute::context context;
        compute::command_queue queue;
        compute::vector<int> vec;
        ushort cardinality;

        bcvec(int dim, const compute::context& context, compute::command_queue &queue ,ushort cardinality): cardinality(cardinality), context(context), queue(queue){
            vec = compute::vector<int>(dim, context);
        }
    };

    //TODO: Refactor partial_result to incapsulate a shared_ptr<map> instead of map. Fix all operators
    template <typename Probability>
    class partial_result {
    private:
        std::shared_ptr<std::map<bn::vertex<Probability>,std::vector<int>>> data;
        uint nsamples;
    public:
        explicit partial_result(uint ns) : nsamples(ns), data(std::make_shared<std::map<bn::vertex<Probability>,std::vector<int>>>()) {}

        partial_result &operator=(const partial_result &other) {
            if (this == &other)
                return *this;
            *(other.data).swap(*this);
            return *this;
        }
        void swap(partial_result &other) {
            (*data).swap(*(other.data));
        }

        partial_result& operator+=(partial_result &other) {
            for (const auto [k,v] : *(other.data)) {
                BAYLIB_ASSERT(other.nsamples == nsamples,
                              "samples vector size must agree",
                              std::runtime_error)

                if ((*data).count(k))
                {
                    for (int i = 0; i<v.size();i++)
                        (*data)[k][i] += v[i];
                }
                else
                    (*data)[k] = v;
            }
            return (*this);
        }

        partial_result operator+(partial_result &other) {
            partial_result new_spr = (*this);
            new_spr+=other;
            return new_spr;
        }
        std::vector<int>& operator[](const bn::vertex<Probability> key) {
            return (*data)[key];
        }

        std::map<bn::vertex<Probability>,std::vector<Probability>> compute_total_result() {
            std::map<bn::vertex<Probability>,std::vector<Probability>> total_result;
            for (auto &[k,v] : *data) {
                Probability tot = 0;
                for (int i: v)
                    tot += Probability(i);
                for (int i: v)
                    total_result[k].push_back(Probability(i)/tot);
            }
            return total_result;
        }

    };


    template <typename Probability>
    class logic_sampling {
        using prob_v = boost::compute::vector<Probability>;

    public:
        explicit logic_sampling(
            const bn::bayesian_network<Probability> &bn,
            const compute::device &device = compute::system::default_device()
        ) 
        : bn(bn)
        , nsamples(0)
        , nthreads(0)
        , niter(0)
        , seed(0)
        {
            /*
		    for (auto v : bn.variables()) {
		        std::cout << v.name() << '\n';
		    }*/
            auto var = bn.variables();
            BAYLIB_ASSERT(std::all_of(var.begin(), var.end(),
                                      [](auto &var){ return bn::cpt_filled_out(var); }),
                          "conditional probability tables must be properly filled to"
                          " run logic_sampling inference algorithm",
                          std::runtime_error)
        }

        std::pair<std::vector<int>, std::shared_ptr<bcvec>> simulate_node(std::vector<Probability> striped_cpt,
                                                 const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                                 int dim = 10000,
                                                 int possible_states = 2);
        std::map<bn::vertex<Probability>,std::vector<Probability>> compute_network_marginal_probabilities(size_t memory, int samples, int n_threads = 1);

    private:
        using uncompressed_partial_result =  std::map< bn::vertex<Probability>, std::shared_future< std::pair<std::vector<int>,std::shared_ptr<bcvec>>>>;
        bn::bayesian_network<Probability> bn;
        compute::context context;
        uint nsamples;
        uint nthreads;
        std::atomic<int> seed;
        uint niter;


        // private members
        std::vector<Probability> accumulate_cpt(std::vector<Probability> striped_cpt, int possible_states);
        std::vector<int> compute_result_general(bcvec &res, compute::command_queue &queue_);
        void calculate_iterations(int n_threads, size_t memory, int samples); // return <n_iterations, samples_in_iter>
        partial_result<Probability> to_partial_result(std::map< bn::vertex<Probability>, std::shared_future<std::pair< std::vector<int>, std::shared_ptr<bcvec> > >>  results);

#if DEBUG_MONTECARLO
        template<typename S>
        void print_vec(compute::vector<S> &vec, const std::string& message, int len);
#endif
    };



    template<typename Probability>
    std::map<bn::vertex<Probability>,std::vector<Probability>> logic_sampling<Probability>::compute_network_marginal_probabilities(
            size_t memory,
            int samples,
            int n_threads) {

        nthreads = n_threads <= 0 ? 1 : n_threads;
        context = compute::context(compute::system::default_device());
        auto vertex_queue = bn::sampling_order(bn);
        seed = 0;
        calculate_iterations(nthreads, memory, samples);

        auto cnmp = [this](
                uncompressed_partial_result results,
                int samples, bn::vertex<Probability> v)-> std::pair<std::vector<int>, std::shared_ptr<bn::bcvec>> {
            int possible_states =  bn[v].states().size();
            std::vector<std::shared_ptr<bcvec>> parents_result;
            std::vector<std::string> parents = bn[v].parents_info.names();
            std::reverse(parents.begin(), parents.end());
            for (auto p : parents) {
                auto par = results[bn.index_of(p)];
                std::shared_ptr<bcvec> res = par.get().second;
                parents_result.push_back(res);
            }
            auto table = bn[v].table().flat();
            return simulate_node( table, parents_result, samples, possible_states);
        };
        //Usato per debuggare senza thread_pool
        auto debug_wrapper = [this,cnmp](uncompressed_partial_result results,
                                         int samples, bn::vertex<Probability> v) {
            std::promise<std::pair<std::vector<int>, std::shared_ptr<bn::bcvec>>> p;
            auto res = cnmp(results,samples,v);
            p.set_value(res);
            return p.get_future();

        };
        partial_result<Probability> pr(nsamples);
        for (int i = 0; i<niter; i++) {
            //thread_pool tp(nthreads);
            uncompressed_partial_result results;
            for(auto v : vertex_queue) {
                //TODO: use result of calculate_iterations instead of samples
                //results[v] = std::move(tp.submit(cnmp, results, samples, v));
                results[v] = debug_wrapper(results, samples, v);
            }
            auto pr_results = to_partial_result(results);
            pr += pr_results;

        }

        return pr.compute_total_result();
    }


#if DEBUG_MONTECARLO
    template<typename Probability>
    template<typename S>
    void logic_sampling<Probability>::print_vec(compute::vector<S> &vec, const std::string& message, int len){
        compute::command_queue queue(context, compute::system::default_device());
        if(len == -1)
            len = vec.size();
        std::vector<S> host_vec(len);
        compute::copy(vec.begin(), vec.begin() + len, host_vec.begin(), queue);
        std::cout << message << ' ';
        for(S el: host_vec)
            std::cout << el << ' ';
        std::cout << '\n';
    }
#endif
    template <typename Probability>
    std::vector<Probability> logic_sampling<Probability>::accumulate_cpt(std::vector<Probability> striped_cpt, int possible_states){
        for(int i = 0 ; i < striped_cpt.size() ; i += possible_states)
            for(int j = 1 ; j < possible_states - 1 ; j++)
                striped_cpt[i + j] += striped_cpt[i + j - 1];
        return striped_cpt;
    }


    /// Node Simulation with GPU parallelization
    /// memory usage in GPU device: dim * (sizeof(Probability) + 3 * sizeof(ushort))) + cpt_len * sizeof(Probability)
    /// \param striped_cpt CPT table in a contiguous format
    /// \param parents_result output of parent nodes, if simulating source leave empty
    /// \param dim number of samples to simulate, it must be consistent with parents simulation
    /// \param possible_states cardinality of the discrete variable to simulate (e.g. 2 if binary variable)
    /// \return shared_ptr to result of simulation, use with other simulations or condense results with compute_result
    template <typename Probability>
    std::pair<std::vector<int>, std::shared_ptr<bcvec>> logic_sampling<Probability>::simulate_node(const std::vector<Probability> striped_cpt,
                                                                          const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                                                          int dim,
                                                                          int possible_states){

        compute::command_queue queue(context, compute::system::default_device());
        std::vector<Probability> striped_cpt_accum = this->accumulate_cpt(striped_cpt, possible_states);
        std::shared_ptr<bcvec> result = std::make_shared<bcvec>(dim, context, queue, possible_states);
        prob_v device_cpt(striped_cpt_accum.size(), context);
        prob_v threshold_vec(dim, context);
        prob_v random_vec(dim, context);
        compute::uniform_real_distribution<Probability> distribution(0, 1);
        compute::vector<int> index_vec(dim, context);
        compute::copy(striped_cpt_accum.begin(), striped_cpt_accum.end(), device_cpt.begin(), queue);

        if(parents_result.empty()){
            compute::fill(index_vec.begin(), index_vec.end(), 0, queue);
        }
        else {
            int coeff = possible_states;
            for (int i = 0; i < parents_result.size(); i++) {
                compute::vector<Probability> copied_parent(parents_result[i]->vec.size(), context);
                compute::copy(parents_result[i]->vec.begin(),parents_result[i]->vec.end(), copied_parent.begin(), queue);
#if DEBUG_MONTECARLO
                print_vec(parents_result[i]->vec, "PARENT", PRINT_N);
#endif
                if (i == 0)
                    compute::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(),
                                       index_vec.begin(), _1 * coeff, queue);
                else
                    compute::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(),
                                       index_vec.begin(),index_vec.begin(),
                                       _1 * coeff + _2, queue);
                coeff *= parents_result[i]->cardinality;
            }
        }
#if DEBUG_MONTECARLO
        print_vec(index_vec, "INDEX", PRINT_N);
        print_vec(device_cpt, "CPT", device_cpt.size());
#endif
        compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
#if DEBUG_MONTECARLO
        print_vec(threshold_vec, "THRESH", PRINT_N);
#endif
        compute::default_random_engine rand_eng(queue, seed);
        seed.fetch_add(1);
        distribution.generate(random_vec.begin(), random_vec.end(), rand_eng, queue);
        compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), result->vec.begin(), _1 > _2, queue);
        for(int i = 0; i + 2 < possible_states ; i++){
            compute::vector<int> temp(dim, context);
            compute::transform(index_vec.begin(), index_vec.end(),
                               index_vec.begin(), _1 + 1, queue);
            compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(),
                            threshold_vec.begin(), queue);
            compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(),
                               temp.begin(), _1 > _2, queue);
            compute::transform(temp.begin(), temp.end(), result->vec.begin(),
                               result->vec.begin(), _1 + _2, result->queue);
        }

#if DEBUG_MONTECARLO
        print_vec(random_vec, "RANDOM", PRINT_N);
        print_vec(result->vec, "RESULT", PRINT_N);
#endif
        std::vector<int> compr_res = compute_result_general(*result, queue);
        return std::pair<std::vector<int>, std::shared_ptr<bcvec>>(compr_res, result);
    }


    /// Accumulate simulation results for general case
    /// \param res result from simulate node
    /// \return vector for witch the i-th element is the number of occurrences of i
    template <typename T>
    std::vector<int> logic_sampling<T>::compute_result_general(bcvec& res, compute::command_queue &queue){
        std::vector<int> acc_res(res.cardinality);
        for (int i = 0; i < res.cardinality; ++i) {
            acc_res[i] = compute::count(res.vec.begin(), res.vec.end(), i, queue);
        }
        return acc_res;
    }

    template<typename Probability>
    void logic_sampling<Probability>::calculate_iterations(int n_threads, size_t memory, int samples) {
        this->nthreads = n_threads;

        uint64_t sample_p = memory/(bn.number_of_variables()*sizeof(Probability)+3*sizeof(cl_ushort)*n_threads) * MEMORY_SLACK;
        if(sample_p < samples){
            this->nsamples = sample_p;
            this->niter = samples / sample_p;
        }
        else {
            this->nsamples = samples;
            this->niter = 1;
        }
        //TODO implement calculation
    }

     template<typename Probability>
     partial_result<Probability> logic_sampling<Probability>::to_partial_result( std::map< bn::vertex<Probability>, std::shared_future<std::pair< std::vector<int>, std::shared_ptr<bcvec> > >> results) {
        partial_result<Probability> pr((uint)nsamples);
        for (auto &[k,v] : results) {
            pr[k] = v.get().first;
        }
        return pr;
    }



} // namespace bn


#endif //BAYLIB_LOGIC_SAMPLING_HPP
