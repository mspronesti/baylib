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
    using bcvec = boost::compute::vector<int>;

    template <typename T>
    class logic_sampling {

    };

    template <typename T=float >
    std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<bcvec>> parents_result, std::promise<bcvec> result);

}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
