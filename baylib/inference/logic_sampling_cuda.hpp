//
// Created by paolo on 21/12/21.
//


#ifndef BAYLIB_LOGIC_SAMPLING_CUDA_HPP
#define BAYLIB_LOGIC_SAMPLING_CUDA_HPP

#include <baylib/inference/abstract_inference_algorithm.hpp>
#include <curand_kernel.h>
#include <baylib/inference/cuda/logic_sampler.cuh>
#include <baylib/tools/gpu/cuda_utils.cuh>
#include <boost/range/adaptors.hpp>
#include <baylib/network/bayesian_utils.hpp>


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
        class logic_sampling_cuda: public inference_algorithm<Network_>{

            typedef Network_ network_type;
            typedef typename network_type::variable_type variable_type;
            typedef typename variable_type::probability_type probability_type;


        public:
            /**
             * logic_sampling_cuda constructor
             * @param bn        : reference to the bayesian network
             * @param nsamples  : number of samples for the simulation
             */
            explicit logic_sampling_cuda(
                    const network_type & bn,
                    ulong nsamples = 1000
            ) : inference_algorithm<Network_>(bn, nsamples){
            }

            /**
             * Inference method
             * @return : marginal distribution
             */
            baylib::marginal_distribution<probability_type> make_inference(){
                cuda_graph<probability_type> graph = make_cuda_graph<probability_type>(this->bn);
                auto vertex_queue = baylib::sampling_order(this->bn);
                std::vector<uint> result_line = simulate(graph, vertex_queue, this->nsamples);
                auto result = reshape_marginal<probability_type>(this->bn, vertex_queue, result_line);
                auto samples = std::accumulate(result[0].begin(), result[0].end(), 0);
                if (samples != 10240){
                    printf("Samples were %d", samples);
                    for(int i=0; i<result.size(); i++){
                        assert(samples==std::accumulate(result[i].begin(), result[i].end(), 0));
                    }
                }
                result.normalize();
                return result;
            }

        };
    }
}



#endif //BAYLIB_LOGIC_SAMPLING_CUDA_HPP
