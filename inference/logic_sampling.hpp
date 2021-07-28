//
// Created by elle on 22/07/21.
//

#ifndef GPUTEST_LOGIC_SAMPLING_HPP
#define GPUTEST_LOGIC_SAMPLING_HPP

#define DEBUG_MONTECARLO 0

#include <vector>
#include <future>
#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

namespace bn {
    using bcvec = boost::compute::vector<int>;
    namespace compute = boost::compute;
    using boost::compute::lambda::_1;
    using boost::compute::lambda::_2;
    template <typename T>
    class logic_sampling {

    private:
        compute::device device;
        compute::context context;
        compute::command_queue queue;
        std::unique_ptr<compute::default_random_engine> rand_eng;
    public:
        template<typename S>
        void print_vec(compute::vector<S> &vec, const std::string& message="", int len=-1){
            if(len == -1)
                len = vec.size();
            std::vector<S> host_vec(len);
            compute::copy(vec.begin(), vec.begin() + len, host_vec.begin(), queue);
            std::cout << message << ' ';
            for(T el: host_vec)
                std::cout << el << ' ';
            std::cout << '\n';
        }

        logic_sampling() {
            this->device = compute::system::default_device();
            this->context = compute::context(device);
            this->queue = compute::command_queue(context, device);
            this->rand_eng = std::make_unique<compute::default_random_engine>(queue);
        }

        explicit logic_sampling(const compute::device &device) {
            this->device = device;
            this->context = compute::context(device);
            this->queue = compute::command_queue(context, device);
            this->rand_eng = std::make_unique<compute::default_random_engine>(queue);
        }


        std::shared_ptr<bcvec> simulate_node(const std::vector<T>& striped_cpt, const std::vector<std::shared_ptr<bcvec>>& parents_result, int dim = 10000, int possible_states = 2){

            if(possible_states != 2)
                std::cerr << "WARNING FEATURE NOT YET IMPLEMENTED";

            std::shared_ptr<bcvec> result = std::make_shared<bcvec>(dim, context);
            compute::vector<T> device_cpt(striped_cpt.size(), context);
            compute::vector<T> threshold_vec(dim, context);
            compute::vector<T> random_vec(dim, context);
            compute::uniform_real_distribution<T> distribution(0, 1);
            bcvec index_vec(dim, context);
            compute::copy(striped_cpt.begin(), striped_cpt.end(), device_cpt.begin(), queue);

            if(parents_result.empty()){
                compute::fill(index_vec.begin(), index_vec.end(), 0, queue);
            }
            else {
                for (int i = 0; i < parents_result.size(); i++) {
                    compute::function<int(int)> gen_index_first = _1 * (1 << (i + 1));
                    compute::function<int(int, int)> gen_index_and_sum = _1 * (1 << (i + 1)) + _2;
#if DEBUG_MONTECARLO
                    print_vec(*parents_result[i], "PARENT", 10);
#endif
                    if (i == 0)
                        compute::transform(parents_result[i]->begin(), parents_result[i]->end(), index_vec.begin(),
                                           gen_index_first, queue);
                    else
                        compute::transform(parents_result[i]->begin(), parents_result[i]->end(), index_vec.begin(),
                                           index_vec.begin(), gen_index_and_sum, queue);
                }
            }

            compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
            distribution.generate(random_vec.begin(), random_vec.end(), *rand_eng, queue);
            compute::function<int(T, T)> compare_binary_prob = _1 >= _2;
            compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), result->begin(), compare_binary_prob, queue);

#if DEBUG_MONTECARLO
            print_vec(index_vec, "INDEX", 10);
            print_vec(threshold_vec, "THRESH", 10);
            print_vec(random_vec, "RANDOM", 10);
            print_vec(*result, "RESULT", 10);
#endif
            return result;
        }

        std::pair<int, int> compute_result_binary(bcvec &res){
            int sum = 0;
            compute::reduce(res.begin(), res.end(), &sum, queue);
            return std::pair<int, int>(res.size() - sum, sum);
        }

        std::vector<int> compute_result_general(bcvec &res, int n_variables){
            std::vector<int> acc_res(n_variables);
            for (int i = 0; i < n_variables; ++i) {
                compute::count(res.begin(), res.end(), acc_res[i], queue);
            }
            return acc_res;
        }

    };



    template <typename T=float>
    std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<bcvec>> parents_result, std::promise<bcvec> result);





}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
