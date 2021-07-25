//
// Created by paolo on 25/07/2021.
//
#include "logic_sampling.hpp"

//std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<bcvec>> parents_result, std::promise<bcvec> result);
template<typename T>
std::pair<int, int>
bn::logic_sampling<T>::simulate_node_agnostic(const std::vector<T>& striped_cpt, const std::vector<bcvec>& parents_result,
                                          bcvec &result){
    if(parents_result.empty() && striped_cpt.size() == 2) {
        int sum = 0;
        compute::bernoulli_distribution<T> distribution(striped_cpt[0]);
        compute::generate(result.begin(), result.end(), distribution);
        compute::reduce(result.begin(), result.end(), &sum, queue);
        return std::pair<int, int>(sum, result.size() - sum);
    }
    else {
        compute::vector<T> device_cpt(striped_cpt.size(), device);
        compute::copy(striped_cpt.begin(), striped_cpt.end(), device_cpt.begin(), queue);
    }

    return std::pair<int, int>();
}



template<typename T>
bn::logic_sampling<T>::logic_sampling() {
    this->device = compute::system::default_device();
    this->context = compute::context(device);
    this->queue = compute::command_queue(context, device);
}
