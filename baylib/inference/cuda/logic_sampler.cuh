//
// Created by paolo on 29/12/21.
//

#ifndef BAYLIB_LOGIC_SAMPLER_CUH
#define BAYLIB_LOGIC_SAMPLER_CUH

#include <baylib/tools/gpu/cuda_utils.cuh>
namespace baylib {
    namespace inference {
        /**
         * Sampling for a bayesian network using cuda optimization
         * @tparam Probability : Type of the cpt entries
         * @param graph        : Bayesian Network
         * @param sim_order    : Order of simulation
         * @param samples      : Number of samples (The algorithm can generate more if needed for optimization purpouse)
         * @return             : Marginal array
         */
        template<typename Probability>
        std::vector<uint>
        simulate(baylib::cuda_graph<Probability> &graph, const std::vector<ulong> &sim_order, uint samples = 10000);
    }
}

#endif //BAYLIB_LOGIC_SAMPLER_CUH
