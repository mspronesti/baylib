//
// Created by paolo on 15/09/21.
//

#ifndef BAYLIB_GPU_UTILS_HPP
#define BAYLIB_GPU_UTILS_HPP

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <utility>

/**
 * @file gpu_utils.hpp
 * @brief utils for using boost::compute
 */


namespace baylib{
    /**
     * Container for gpu vectors with built in auto release of the memory after set number of uses
     */
    struct bcvec{

        bcvec() = default;

        bcvec(uint dimension, uint cardinality, boost::compute::context &context):
        state(dimension, context), cardinality(cardinality){}

        bcvec(boost::compute::vector<int> state, uint cardinality, uint uses):
        state(std::move(state)), cardinality(cardinality), isevidence(false), evidence_state(0), uses(uses == 0 ? 1 : uses){}

        bcvec(uint evidence_state, uint cardinality):
        isevidence(true), cardinality(cardinality), evidence_state(evidence_state), uses(0) {}

        void add_use(){
            uses--;
            if(!uses)
                state.clear();
        }

    private:
        uint uses{};

    public:
        boost::compute::vector<int> state;
        uint cardinality{};
        bool isevidence{};
        uint evidence_state{};

    };
}


#endif //BAYLIB_GPU_UTILS_HPP