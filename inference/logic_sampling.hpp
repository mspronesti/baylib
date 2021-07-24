//
// Created by elle on 22/07/21.
//

#ifndef GPUTEST_LOGIC_SAMPLING_HPP
#define GPUTEST_LOGIC_SAMPLING_HPP

#define BOOST_COMPUTE_THREAD_SAFE

#include <vector>
#include <future>
#include <boost/compute.hpp>

namespace bn {
    template <typename T>
    class logic_sampling {

    };
    template <typename T=float >
    std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<boost::compute::vector<int>>> parents_result, std::promise<boost::compute::vector<int>> result);

}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
