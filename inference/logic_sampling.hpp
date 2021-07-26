//
// Created by elle on 22/07/21.
//

#ifndef GPUTEST_LOGIC_SAMPLING_HPP
#define GPUTEST_LOGIC_SAMPLING_HPP

#define BOOST_COMPUTE_THREAD_SAFE

#include <vector>
#include <future>
#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

namespace bn {
    using bcvec = boost::compute::vector<int>;
    namespace compute = boost::compute;
    template <typename T>
    class logic_sampling {

    private:
        compute::device device;
        compute::context context;
        compute::command_queue queue;
        std::unique_ptr<compute::default_random_engine> rand_eng;
    public:
        template<typename S>
        void print_vec(compute::vector<S> &vec){
            std::cout << "HERE\n";
            std::vector<S> host_vec(vec.size());
            compute::copy(vec.begin(), vec.end(), host_vec.begin(), queue);
            for(T el: vec){
                std::cout << el << ' ';
            }
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

        std::pair<int, int>simulate_node_agnostic_debug(const std::vector<T>& striped_cpt, int possible_states, const std::vector<std::vector<int>>& parents_result, int dim=10){
            std::vector<bcvec> gpu_par_result(parents_result.size());
            bcvec result(dim, context);
            std::transform(parents_result.begin(), parents_result.end(), gpu_par_result.begin(), [this](std::vector<int> par)->bcvec{
                bcvec vec(par.size(), context);
                boost::compute::copy(par.begin(), par.end(), vec.begin(), queue);
                return vec;
            });
            return simulate_node_agnostic(striped_cpt, possible_states, gpu_par_result, result);
        }

        std::pair<int, int> simulate_node_agnostic(const std::vector<T>& striped_cpt, int possible_states, const std::vector<bcvec>& parents_result, bcvec& result){
            if(parents_result.empty() && possible_states == 2) {
                // Simple case with binary probabilities and no parents, solvable with simple bernoulli distribution
                int sum = 0;
                compute::bernoulli_distribution<T> distribution(striped_cpt[0]);
                distribution.generate(result.begin(), result.end(), *rand_eng, queue);
                compute::reduce(result.begin(), result.end(), &sum, queue);
                return std::pair<int, int>(sum, result.size() - sum);

            }
            else {
                // Simple case with binary probabilities
                compute::vector<T> device_cpt(striped_cpt.size(), context);
                compute::vector<T> threshold_vec(result.size(), context);
                bcvec index_vec(result.size(), context);
                compute::copy(striped_cpt.begin(), striped_cpt.end(), device_cpt.begin(), queue);
                for (int i = 0 ; i < parents_result.size() ; i++) {
                    compute::function<int(int)> gen_index_first = compute::lambda::_1 * (1 << (i + 1));
                    compute::function<int(int, int)> gen_index_and_sum = compute::lambda::_1 * (1 << (i + 1)) + compute::lambda::_2;
                    if(i == 0)
                        compute::transform(parents_result[i].begin(), parents_result[i].end(), index_vec.begin(), gen_index_first, queue);
                    else
                        compute::transform(parents_result[i].begin(), parents_result[i].end(), index_vec.begin(), index_vec.begin(), gen_index_and_sum, queue);
                }
                //compute::function<T(int)> get_threshold = device_cpt.begin() + compute::lambda::_1;
                //compute::transform(index_vec.begin(), index_vec.end(), threshold_vec.begin(), get_threshold, queue);
                print_vec(index_vec);
                compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
                print_vec(device_cpt);
                print_vec(threshold_vec);
                compute::vector<T> random_vec(result.size(), context);
                compute::uniform_real_distribution<T> distribution(0.0f, 1.0f);
                distribution.generate(random_vec.begin(), random_vec.end(), *rand_eng, queue);
            }

            return std::pair<int, int>();
        }

    };



    template <typename T=float>
    std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<bcvec>> parents_result, std::promise<bcvec> result);





}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
