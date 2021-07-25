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

    public:
        logic_sampling();
        std::pair<int, int> simulate_node_agnostic(const std::vector<T>& striped_cpt, const std::vector<bcvec>& parents_result, bcvec& result);

    };



    template <typename T=float>
    std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<bcvec>> parents_result, std::promise<bcvec> result);





}


#endif //GPUTEST_LOGIC_SAMPLING_HPP
