//
// Created by paolo on 13/02/22.
//

#ifndef BAYLIB_LIKELIHOOD_WEIGHTING_CUDA_HPP
#define BAYLIB_LIKELIHOOD_WEIGHTING_CUDA_HPP

#include <baylib/inference/abstract_inference_algorithm.hpp>
#include <baylib/inference/cuda/samplers_cuda.cuh>
#include <baylib/tools/gpu/cuda_utils.cuh>
#include <baylib/network/bayesian_utils.hpp>

//! \file likelihood_weighting_cuda.hpp
//! \brief Likelihood Weighting Sampling implementation with cuda optimization

namespace baylib {
    namespace inference {
        /**
        * This class represents the Logic Sampling approximate
        * inference algorithm for discrete Bayesian Networks.
        * The implementation uses cuda to exploit GPGPU optimization:
        *   1. sort nodes in topological order
        *   2. upload all the graph structure into global memory
        *   3. launch a grid of threads were each one makes a simulation of the whole network
        *   5. accumulate the marginal results and estimate the marginal distribution
        * @tparam Network_ : the type of bayesian network (must inherit from baylib::bayesian_net)
        **/
        template <BNetDerived Network_>
        class likelihood_weighting_cuda: public inference_algorithm<Network_>{

            typedef Network_ network_type;
            typedef typename network_type::variable_type variable_type;
            typedef typename variable_type::probability_type probability_type;


        public:
            /**
             * logic_sampling_cuda constructor
             * @param bn        : reference to the bayesian network
             * @param nsamples  : number of samples for the simulation
             */
            explicit likelihood_weighting_cuda(
                    const network_type & bn,
                    ulong nsamples = 1000
            ) : inference_algorithm<Network_>(bn, nsamples){
            }

            /**
             * Inference method
             * @return : marginal distribution
             */
            baylib::marginal_distribution<probability_type> make_inference(){
                cuda_graph_adapter<probability_type> graph = make_cuda_graph_revised<probability_type>(this->bn);
                auto vertex_queue = baylib::sampling_order(this->bn);
                baylib::marginal_distribution<probability_type> result(this->bn.begin(), this->bn.end());
                if(evidence_presence(this->bn)) {
                    std::vector<float> result_line = likelihood_weighting_sampler(
                            graph, vertex_queue, this->nsamples, this->seed
                            );
                    result = reshape_marginal<probability_type>(this->bn, vertex_queue, result_line);
                }
                else{
                    std::vector<uint> result_line = logic_sampler(
                            graph,vertex_queue,this->nsamples,false,this->seed
                            );
                    result = reshape_marginal<probability_type>(this->bn, vertex_queue, result_line);
                }
                result.normalize();
                return result;
            }

        };
    }
}

#endif //BAYLIB_LIKELIHOOD_WEIGHTING_CUDA_HPP
