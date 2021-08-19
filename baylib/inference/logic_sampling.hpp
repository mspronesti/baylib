//
// Created by elle on 22/07/21.
//

#ifndef BAYLIB_LOGIC_SAMPLING_HPP
#define BAYLIB_LOGIC_SAMPLING_HPP

#define DEBUG_MONTECARLO 0

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
        compute::vector<int> vec;
        ushort cardinality;
        bcvec(int dim, const compute::context& context, ushort cardinality): cardinality(cardinality){
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
        explicit partial_result(int ns) : nsamples(ns), data(std::make_shared<std::map<bn::vertex<Probability>,std::vector<int>>>()) {}

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
            Probability nsamples_p = Probability(nsamples);
            for (auto &[k,v] : *data) {
                for (int i: v)
                    total_result[k].push_back(Probability(i)/nsamples_p);
            }
            return total_result;
        }

	};


    template <typename Probability>
    class logic_sampling {
        using prob_v = boost::compute::vector<Probability>;

    public:
        explicit logic_sampling(const bn::bayesian_network<Probability> &bn,
                const compute::device &device = compute::system::default_device())
                : bn(bn), device(device), nsamples(0), nthreads(0), niter(0)
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
            this->context = compute::context(device);
            this->queue = compute::command_queue(context, device);
            this->rand_eng = std::make_unique<compute::default_random_engine>(queue);
		}

        std::shared_ptr<bn::bcvec> simulate_node(const std::vector<Probability>& striped_cpt,
                                                 const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                                 int dim = 10000,
                                                 int possible_states = 2);
        std::map<bn::vertex<Probability>,std::vector<Probability>> compute_network_marginal_probabilities(size_t memory, int samples, int n_threads = 1);

    private:
		compute::device device;
        compute::context context;
        compute::command_queue queue;
        std::unique_ptr<compute::default_random_engine> rand_eng;
		bn::bayesian_network<Probability> bn;
		uint nsamples;
		uint nthreads;
		uint niter;
		std::mutex lck;

        // private members
        std::vector<Probability> accumulate_cpt(std::vector<Probability> striped_cpt, int possible_states);
        std::pair<int, int> compute_result_binary(bcvec &res);
        std::vector<int> compute_result_general(bcvec &res);
        void calculate_iterations(int n_threads, size_t memory, int samples); // return <n_iterations, samples_in_iter>
        partial_result<Probability> to_partial_result(std::shared_ptr< std::map< bn::vertex<Probability>, std::shared_future< std::shared_ptr<bcvec> > > > results);
        template<typename S>
        void print_vec(compute::vector<S> &vec, const std::string& message, int len);
        };



    template<typename Probability>
    std::map<bn::vertex<Probability>,std::vector<Probability>> logic_sampling<Probability>::compute_network_marginal_probabilities(
            size_t memory,
            int samples,
            int n_threads) {
        using uncompressed_partial_result = std::shared_ptr< std::map< bn::vertex<Probability>, std::shared_future< std::shared_ptr<bcvec>>>>;
        nthreads = n_threads <= 0 ? 1 : n_threads;
        thread_pool tp(nthreads);
        auto vertex_queue = bn::sampling_order(bn);

        calculate_iterations(nthreads, memory, samples);

        auto cnmp = [this](
                uncompressed_partial_result results,
                int samples, bn::vertex<Probability> v)-> std::shared_ptr<bn::bcvec> {
            int possible_states =  bn[v].states().size();
            std::vector<std::shared_ptr<bcvec>> parents_result;
            std::vector<std::string> parents = bn[v].parents_names();
            std::reverse(parents.begin(), parents.end());
            for (auto p : parents) {
                std::scoped_lock<std::mutex> lock(lck);
                parents_result.push_back((*results)[bn.index_of(p)].get());
            }
            return simulate_node( bn[v].table().flat(), parents_result, samples, possible_states);
        };
        //Usato per debuggare senza thread_pool
        auto debug_wrapper = [this,cnmp](uncompressed_partial_result results,
                                         int samples, bn::vertex<Probability> v) {
            std::promise<std::shared_ptr<bn::bcvec>> p;
            auto res = cnmp(results,samples,v);
            p.set_value(res);
            return p.get_future();

        };
        partial_result<Probability> pr(nsamples);
        for (int i = 0; i<niter; i++) {
            uncompressed_partial_result results = std::make_shared<std::map< bn::vertex<Probability>, std::shared_future< std::shared_ptr<bcvec> > >>();
            for (bn::vertex<Probability> v: vertex_queue) {
                //TODO: use result of calculate_iterations instead of samples
                (*results)[v] = tp.submit(cnmp, results, samples, v);
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
    std::shared_ptr<bn::bcvec> logic_sampling<Probability>::simulate_node(const std::vector<Probability>& striped_cpt,
                                                                          const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                                                          int dim,
                                                                          int possible_states){

        std::vector<Probability> striped_cpt_accum = this->accumulate_cpt(striped_cpt, possible_states);
        std::shared_ptr<bcvec> result = std::make_shared<bcvec>(dim, context, possible_states);
        prob_v device_cpt(striped_cpt.size(), context);
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
#if DEBUG_MONTECARLO
                print_vec(parents_result[i]->vec, "PARENT", 10);
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
        compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
#if DEBUG_MONTECARLO
        print_vec(index_vec, "INDEX", 10);
        print_vec(threshold_vec, "THRESH", 10);
#endif
        distribution.generate(random_vec.begin(), random_vec.end(), *rand_eng, queue);
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
                               result->vec.begin(), _1 + _2, queue);
        }

#if DEBUG_MONTECARLO
        print_vec(random_vec, "RANDOM", 10);
        print_vec(result->vec, "RESULT", 10);
#endif
        return result;
    }

    /// Accumulate simulation results for binary case
    /// \param res result from simulate_node
    /// \return pair<Occurrences of 0, Occurrences of 1>
    template <typename T>
    std::pair<int, int> logic_sampling<T>::compute_result_binary(bcvec &res){
        int sum = 0;
        compute::reduce(res.vec.begin(), res.vec.end(), &sum, queue);
        return std::pair<int, int>(res.vec.size() - sum, sum);
    }

    /// Accumulate simulation results for general case
    /// \param res result from simulate node
    /// \return vector for witch the i-th element is the number of occurrences of i
    template <typename T>
    std::vector<int> logic_sampling<T>::compute_result_general(bcvec &res){
        std::vector<int> acc_res(res.cardinality);
        for (int i = 0; i < res.cardinality; ++i) {
            acc_res[i] = compute::count(res.vec.begin(), res.vec.end(), i, queue);
        }
        return acc_res;
    }

    template<typename Probability>
    void logic_sampling<Probability>::calculate_iterations(int n_threads, size_t memory, int samples) {
        this->nthreads = n_threads;
        this->nsamples = samples;
        this->niter = 1;
        //TODO implement calculation
    }

    template<typename Probability>
    partial_result<Probability> logic_sampling<Probability>::to_partial_result(std::shared_ptr< std::map< bn::vertex<Probability>, std::shared_future< std::shared_ptr<bcvec> > > > results) {
        partial_result<Probability> pr(nsamples);
        for (auto &[k,v] : (*results)) {
            pr[k] = compute_result_general(*(v.get()));
        }
        return pr;
    }



} // namespace bn


#endif //BAYLIB_LOGIC_SAMPLING_HPP
