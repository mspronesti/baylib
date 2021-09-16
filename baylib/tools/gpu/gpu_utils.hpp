//
// Created by paolo on 15/09/21.
//

#ifndef BAYLIB_GPU_UTILS_HPP
#define BAYLIB_GPU_UTILS_HPP

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

namespace bn{
    /**
     * Container for gpu vectors with built in auto release of the memory after set number of uses
     */
    struct bcvec{

        bcvec() = default;

        bcvec(boost::compute::vector<cl_short> state, uint cardinality, uint uses):
        state(state), cardinality(cardinality), isevidence(false), evidence_state(0), uses(uses == 0 ? 1 : uses){}

        bcvec(uint evidence_state, uint cardinality):
        isevidence(true), cardinality(cardinality), evidence_state(evidence_state), uses(0) {}


        bool set_use(){
            uses--;
            if(uses == 0){
                state.reset();
            }
            return uses == 0;
        }

        boost::compute::vector<cl_short>& get_states(){
            BAYLIB_ASSERT(uses > 0 , "Cannot use this vector anymore", std::logic_error);
            BAYLIB_ASSERT(!isevidence , "Cannot get vector of evidences", std::logic_error);
            return state.value();
        }

    private:
        std::optional<boost::compute::vector<cl_short>> state;
        uint uses{};

    public:
        uint cardinality{};
        bool isevidence{};
        uint evidence_state{};

    };
}


#endif //BAYLIB_GPU_UTILS_HPP
