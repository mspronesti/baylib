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

        std::pair<int, int> simulate_node_agnostic(const std::vector<T>& striped_cpt, const std::vector<bcvec>& parents_result, bcvec& result){
            if(parents_result.empty() && striped_cpt.size() == 2) {
                int sum = 0;
                compute::bernoulli_distribution<T> distribution(striped_cpt[0]);
                distribution.generate(result.begin(), result.end(), *this->rand_eng, queue);
                compute::reduce(result.begin(), result.end(), &sum, queue);
                return std::pair<int, int>(sum, result.size() - sum);
            }
            else {
                compute::vector<T> device_cpt(striped_cpt.size(), context);
                compute::copy(striped_cpt.begin(), striped_cpt.end(), device_cpt.begin(), queue);
            }

            return std::pair<int, int>();
        }

    };



    template <typename T=float>
    std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<bcvec>> parents_result, std::promise<bcvec> result);





}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
