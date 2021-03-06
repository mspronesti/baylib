//
// Created by elle on 22/07/21.
//

#ifndef BAYLIB_LOGIC_SAMPLING_OPENCL_HPP
#define BAYLIB_LOGIC_SAMPLING_OPENCL_HPP

#define CL_TARGET_OPENCL_VERSION 220


#include <vector>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include "baylib/probability/condition_factory.hpp"

#include "baylib/inference/abstract_inference_algorithm.hpp"
#include "vectorized_inference_opencl.hpp"

//! \file logic_sampling_opencl.hpp
//! \brief Logic Sampling implementation with opencl optimization

namespace baylib {
    namespace inference{
        namespace compute = boost::compute;
        using boost::compute::lambda::_1;
        using boost::compute::lambda::_2;


        /**
        * This class represents the Logic Sampling approximate
        * inference algorithm for discrete Bayesian Networks.
        * The implementation uses boost::compute to exploit GPGPU optimization.
        * All samples are simulated in parallel, for this reason the maximum memory usage
        * tolerable must be specified to avoid filling up the memory of the device in case of large
        * number of samples. The algorithm main steps are :
        *   1. sort nodes in topological order
        *   2. simulate each source node
        *   3. simulate each child node given the parents previous simulation
        *   4. throw out each simulation that doesn't comply with observed evidences
        *   5. estimate the marginal distribution from the valid simulations
         * @tparam Network_ : the type of bayesian network (must inherit from baylib::bayesian_net)
         * @tparam Generator_ : the type of random generator
         *                  (default Mersenne Twister pseudo-random generator)
        **/
        template <
                BNetDerived Network_,
                typename Generator_ = std::mt19937
                >
        class logic_sampling_opencl : public baylib::inference::vectorized_inference_algorithm<Network_>
        {
            using typename vectorized_inference_algorithm<Network_>::probability_type;
            using vectorized_inference_algorithm<Network_>::bn;
            using prob_v = boost::compute::vector<probability_type>;
            typedef Network_ network_type;
        public:

            logic_sampling_opencl(
                    const network_type &bn,
                    ulong samples,
                    size_t memory,
                    uint seed = 0,
                    const compute::device &device = compute::system::default_device()
            )
            : vectorized_inference_algorithm<Network_>(bn, samples, memory, seed, device)
            { }

            marginal_distribution<probability_type> make_inference () override
            {
                BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                          [this](auto &var){ return baylib::cpt_filled_out(bn, var.id()); }),
                              "conditional probability tables must be properly filled to"
                              " run logic_sampling_opencl inference algorithm",
                              std::runtime_error);

                auto [iter_samples, niter] = this->calculate_iterations();
                auto vertex_queue = baylib::sampling_order(bn);
                marginal_distribution<probability_type> marginal_result(bn.begin(), bn.end());
                for (ulong i = 0; i< niter; i++) {

                    std::vector<bcvec> result_container(vertex_queue.size());
                    marginal_distribution<probability_type> temp(bn.begin(), bn.end());
                    compute::vector<int> valid_evidence_vec(this->nsamples, true, this->queue);

                    for(ulong v : vertex_queue) {

                        std::vector<bcvec*> parents_result;

                        // Build parents result vector in the correct order
                        auto parents = bn.parents_of(v);
                        //std::reverse(parents.begin(), parents.end());

                        for (auto p : parents) {
                            parents_result.push_back(&result_container[p]);
                        }

                        result_container[v] = this->simulate_node(v, bn[v].table(), parents_result, iter_samples);

                        if(bn[v].is_evidence()){
                            compute::transform( result_container[v].state.begin()
                                               ,result_container[v].state.end()
                                               ,valid_evidence_vec.begin()
                                               ,valid_evidence_vec.begin()
                                               ,(_1 == bn[v].evidence_state()) && _2
                                               ,this->queue);
                        }
                    }

                    for(ulong v : vertex_queue) {
                        auto accumulated_result = compute_result_general(result_container[v], valid_evidence_vec);
                        for(int ix=0; ix< accumulated_result.size(); ix++)
                            marginal_result[v][ix] += accumulated_result[ix];
                    }
                }
                marginal_result.normalize();
                return marginal_result;
            }


        private:

            /**
            * Accumulate simulation results for general case
            * @param res result from simulate node
            * @return vector for witch the i-th element is the number of occurrences of i
            **/
            std::vector<ulong> compute_result_general(
                    bcvec& res,
                    compute::vector<int>& valid
            )
            {
                compute::transform( res.state.begin()
                                   ,res.state.end()
                                   ,valid.begin()
                                   ,res.state.begin()
                                   ,(_1+1)*_2
                                   ,this->queue);
                std::vector<ulong> acc_res(res.cardinality);
                for (baylib::state_t i = 0; i < res.cardinality; ++i) {
                    acc_res[i] = compute::count(res.state.begin(), res.state.end(), i+1, this->queue);
                }
                return acc_res;
            }
        };
    } // namespace inference
} // namespace baylib


#endif //BAYLIB_LOGIC_SAMPLING_OPENCL_HPP