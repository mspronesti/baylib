//
// Created by elle on 22/07/21.
//

#ifndef BAYESIAN_INFERRER_LOGIC_SAMPLING_HPP
#define BAYESIAN_INFERRER_SAMPLING_HPP

#define DEBUG_MONTECARLO 0

#include <vector>
#include <future>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

#include <baylib/network/bayesian_network.hpp>

namespace bn {
    namespace compute = boost::compute;
	using boost::compute::lambda::_1;
    using boost::compute::lambda::_2;

	struct bcvec {
        compute::vector<int> vec;
        int cardinality;
        bcvec(int dim, const compute::context& context, int cardinality): cardinality(cardinality){
            vec = compute::vector<int>(dim, context);
        }
    };

    template <typename Probability>
    class logic_sampling {
        using prob_v = boost::compute::vector<Probability>;

    public:
		explicit logic_sampling(const std::shared_ptr<bn::bayesian_network<Probability>> &bn,
                const compute::device &device = compute::system::default_device())
                : bn(bn), device(device)
        {
            this->context = compute::context(device);
            this->queue = compute::command_queue(context, device);
            this->rand_eng = std::make_unique<compute::default_random_engine>(queue);
		}

        std::shared_ptr<bn::bcvec> simulate_node(const std::vector<Probability>& striped_cpt,
                                                 const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                                 int dim = 10000,
                                                 int possible_states = 2);

    private:
		compute::device device;
        compute::context context;
        compute::command_queue queue;
        std::unique_ptr<compute::default_random_engine> rand_eng;
		std::shared_ptr<bn::bayesian_network<Probability>> bn;

        // private members
        std::vector<Probability> accumulate_cpt(std::vector<Probability> striped_cpt, int possible_states);
        std::pair<int, int> compute_result_binary(bcvec &res);
        std::vector<int> compute_result_general(bcvec &res);
        std::pair<int, int> calculate_iterations(int nthreads, size_t memory, int samples); // return <n_iterations, samples_in_iter>
    };


#if DEBUG_MONTECARLO
    template<typename S>
    void logic_sampling<Probability>::print_vec(compute::vector<S> &vec, const std::string& message="", int len=-1){
        if(len == -1)
            len = vec.size();
        std::vector<S> host_vec(len);
        compute::copy(vec.begin(), vec.begin() + len, host_vec.begin(), queue);
        std::cout << message << ' ';
        for(Probability el: host_vec)
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
    /// \param striped_cpt CPT table in a contiguous format
    /// \param parents_result output of parent nodes, if simulating source leave empty
    /// \param dim number of samples to simulate, it must be consistent with parents simulation
    /// \param possible_states cardinality of the discrete variable to simulate (e.g. 2 if binary variable)
    /// \return shared_ptr to result of simulation, use with other simulations or condense results with compute_result
    template <typename T>
    std::shared_ptr<bn::bcvec> logic_sampling<T>::simulate_node(const std::vector<T>& striped_cpt,
                                         const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                         int dim,
                                         int possible_states){


        std::vector<T> striped_cpt_accum = this->accumulate_cpt(striped_cpt, possible_states);
        std::shared_ptr<bcvec> result = std::make_shared<bcvec>(dim, context, possible_states);
        prob_v device_cpt(striped_cpt.size(), context);
        prob_v threshold_vec(dim, context);
        prob_v random_vec(dim, context);
        compute::uniform_real_distribution<T> distribution(0, 1);
        compute::vector<int> index_vec(dim, context);

        compute::copy(striped_cpt_accum.begin(), striped_cpt_accum.end(), device_cpt.begin(), queue);

        if(parents_result.empty()){
            compute::fill(index_vec.begin(), index_vec.end(), 0, queue);
        }
        else {
            int coeff = possible_states;
            for (int i = 0; i < parents_result.size(); i++) {
#if DEBUG_MONTECARLO
                print_vec(*parents_result[i], "PARENT", 10);
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
        print_vec(*result, "RESULT", 10);
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
    std::pair<int, int> logic_sampling<Probability>::calculate_iterations(int nthreads, size_t memory, int samples) {
        return std::pair<int, int>(1, samples); //TODO implement calculation
    }


} // namespace bn


#endif //BAYESIAN_INFERRER_LOGIC_SAMPLING_HPP
