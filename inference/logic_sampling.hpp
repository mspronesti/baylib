//
// Created by elle on 22/07/21.
//

#ifndef GPUTEST_LOGIC_SAMPLING_HPP
#define GPUTEST_LOGIC_SAMPLING_HPP

#define DEBUG_MONTECARLO 0
#define BOOST_COMPUTE_THREAD_SAFE

#include <vector>
#include <future>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/core.hpp>



namespace bn {
    namespace cpt = boost::compute;
    using boost::compute::lambda::_1;
    using boost::compute::lambda::_2;

    struct bcvec {
        cpt::vector<int> vec;
        int cardinality;
        bcvec(int dim, const cpt::context& context, int cardinality): cardinality(cardinality){
            vec = cpt::vector<int>(dim, context);
        }
    };

    template <typename T>
    class logic_sampling {
    private:
        cpt::device device;
        cpt::context context;
        cpt::command_queue queue;
        std::unique_ptr<cpt::default_random_engine> rand_eng;

#if DEBUG_MONTECARLO
        template<typename S>
        void print_vec(cpt::vector<S> &vec, const std::string& message="", int len=-1){
            if(len == -1)
                len = vec.size();
            std::vector<S> host_vec(len);
            cpt::copy(vec.begin(), vec.begin() + len, host_vec.begin(), queue);
            std::cout << message << ' ';
            for(T el: host_vec)
                std::cout << el << ' ';
            std::cout << '\n';
        }
#endif
        std::vector<T> accumulate_cpt(std::vector<T> striped_cpt, int possible_states){
            for(int i = 0 ; i < striped_cpt.size() ; i += possible_states)
                for(int j = 1 ; j < possible_states - 1 ; j++)
                    striped_cpt[i + j] += striped_cpt[i + j - 1];
            return striped_cpt;
        }

    public:

        logic_sampling() {
            this->device = cpt::system::default_device();
            this->context = cpt::context(device);
            this->queue = cpt::command_queue(context, device);
            this->rand_eng = std::make_unique<cpt::default_random_engine>(queue);
        }

        explicit logic_sampling(const cpt::device &device) {
            this->device = device;
            this->context = cpt::context(device);
            this->queue = cpt::command_queue(context, device);
            this->rand_eng = std::make_unique<cpt::default_random_engine>(queue);
        }



        /// Node Simulation with GPU parallelization
        /// \param striped_cpt CPT table in a contiguous format
        /// \param parents_result output of parent nodes, if simulating source leave empty
        /// \param dim number of samples to simulate, it must be consistent with parents simulation
        /// \param possible_states cardinality of the discrete variable to simulate (e.g. 2 if binary variable)
        /// \return shared_ptr to result of simulation, use with other simulations or condense results with compute_result
        std::shared_ptr<bcvec> simulate_node(const std::vector<T>& striped_cpt,
                                             const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                             int dim = 10000,
                                             int possible_states = 2){


            std::vector<T> striped_cpt_accum = this->accumulate_cpt(striped_cpt, possible_states);
            std::shared_ptr<bcvec> result = std::make_shared<bcvec>(dim, context, possible_states);
            cpt::vector<T> device_cpt(striped_cpt.size(), context);
            cpt::vector<T> threshold_vec(dim, context);
            cpt::vector<T> random_vec(dim, context);
            cpt::uniform_real_distribution<T> distribution(0, 1);
            cpt::vector<int> index_vec(dim, context);

            cpt::copy(striped_cpt_accum.begin(), striped_cpt_accum.end(), device_cpt.begin(), queue);

            if(parents_result.empty()){
                cpt::fill(index_vec.begin(), index_vec.end(), 0, queue);
            }
            else {
                int coeff = possible_states;
                for (int i = 0; i < parents_result.size(); i++) {
#if DEBUG_MONTECARLO
                    print_vec(*parents_result[i], "PARENT", 10);
#endif
                    if (i == 0)
                        cpt::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(), index_vec.begin(),
                                           _1 * coeff, queue);
                    else
                        cpt::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(), index_vec.begin(),
                                       index_vec.begin(), _1 * coeff + _2, queue);
                    coeff *= parents_result[i]->cardinality;
                }
            }
            cpt::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
#if DEBUG_MONTECARLO
            print_vec(index_vec, "INDEX", 10);
            print_vec(threshold_vec, "THRESH", 10);
#endif
            distribution.generate(random_vec.begin(), random_vec.end(), *rand_eng, queue);
            cpt::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), result->vec.begin(), _1 > _2, queue);
            for(int i = 0; i + 2 < possible_states ; i++){
                cpt::vector<int> temp(dim, context);
                cpt::transform(index_vec.begin(), index_vec.end(), index_vec.begin(), _1 + 1, queue);
                cpt::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
                cpt::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), temp.begin(), _1 > _2, queue);
                cpt::transform(temp.begin(), temp.end(), result->vec.begin(), result->vec.begin(), _1 + _2, queue);
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
        std::pair<int, int> compute_result_binary(bcvec &res){
            int sum = 0;
            cpt::reduce(res.vec.begin(), res.vec.end(), &sum, queue);
            return std::pair<int, int>(res.vec.size() - sum, sum);
        }

        /// Accumulate simulation results for general case
        /// \param res result from simulate node
        /// \return vector for witch the i-th element is the number of occurrences of i
        std::vector<int> compute_result_general(bcvec &res){
            std::vector<int> acc_res(res.cardinality);
            for (int i = 0; i < res.cardinality; ++i) {
                acc_res[i] = cpt::count(res.vec.begin(), res.vec.end(), i, queue);
            }
            return acc_res;
        }

    };

}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
