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



namespace bn {
    //using bcvec = boost::compute::vector<int>;
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

        std::vector<T> accumulate_cpt(std::vector<T> striped_cpt, int possible_states){
            for(int i = 0 ; i < striped_cpt.size() ; i += possible_states)
                for(int j = 1 ; j < possible_states - 1 ; j++)
                    striped_cpt[i + j] += striped_cpt[i + j - 1];
            return striped_cpt;
        }

        std::shared_ptr<bcvec> simulate_node(const std::vector<T>& striped_cpt,
                                             const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                             int dim = 10000,
                                             int possible_states = 2){


            std::vector<T> striped_cpt_accum = this->accumulate_cpt(striped_cpt, possible_states);
            std::shared_ptr<bcvec> result = std::make_shared<bcvec>(dim, context, possible_states);
            compute::vector<T> device_cpt(striped_cpt.size(), context);
            compute::vector<T> threshold_vec(dim, context);
            compute::vector<T> random_vec(dim, context);
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
                        compute::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(), index_vec.begin(),
                                           _1 * coeff, queue);
                    else
                        compute::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(), index_vec.begin(),
                                           index_vec.begin(), _1 * coeff + _2, queue);
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
                compute::transform(index_vec.begin(), index_vec.end(), index_vec.begin(), _1 + 1, queue);
                compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
                compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), temp.begin(), _1 > _2, queue);
                compute::transform(temp.begin(), temp.end(), result->vec.begin(), result->vec.begin(), _1 +_2, queue);
            }

#if DEBUG_MONTECARLO
            print_vec(random_vec, "RANDOM", 10);
            print_vec(*result, "RESULT", 10);
#endif
            return result;
        }

        std::pair<int, int> compute_result_binary(bcvec &res){
            int sum = 0;
            compute::reduce(res.vec.begin(), res.vec.end(), &sum, queue);
            return std::pair<int, int>(res.vec.size() - sum, sum);
        }

        std::vector<int> compute_result_general(bcvec &res, int n_variables){
            std::vector<int> acc_res(n_variables);
            for (int i = 0; i < n_variables; ++i) {
                acc_res[i] = compute::count(res.vec.begin(), res.vec.end(), i, queue);
            }
            return acc_res;
        }

    };

}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
