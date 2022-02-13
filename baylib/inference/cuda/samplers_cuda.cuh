//
// Created by paolo on 29/12/21.
//

#ifndef BAYLIB_SAMPLERS_CUDA_CUH
#define BAYLIB_SAMPLERS_CUDA_CUH

#include <baylib/tools/gpu/cuda_utils.cuh>


//! \file samplers_cuda.cuh
//! \brief header for sampling algorithm with cuda optimization

namespace baylib {
    namespace inference {
        /**
         * Logic Sampling for bayesian networks using cuda optimization
         * @tparam Probability : Type of the cpt entries
         * @param graph        : Bayesian Network
         * @param sim_order    : Order of simulation
         * @param samples      : Number of samples (The algorithm can generate more if needed for optimization purposes)
         * @param evidence     : hint for the algorithm if he has to check for evidences
         * @return             : Marginal array
         */
        template<typename Probability>
        std::vector<uint>
        logic_sampler(baylib::cuda_graph<Probability> &graph, const std::vector<ulong> &sim_order, uint samples = 10000, bool evidence=true);

        /**
         * Likelihood Sampling for bayesian networks using cuda optimization
         * @tparam Probability : Type of the cpt entries
         * @param graph        : Bayesian Network
         * @param sim_order    : Order of simulation
         * @param samples      : Number of samples (The algorithm can generate more if needed for optimization purposes)
         * @return             : Marginal array
         */
        template<typename Probability>
        std::vector<float>
        likelihood_weighting_sampler(baylib::cuda_graph<Probability> &graph, const std::vector<ulong> &sim_order, uint samples = 10000);

    }
}

#endif //BAYLIB_SAMPLERS_CUDA_CUH
