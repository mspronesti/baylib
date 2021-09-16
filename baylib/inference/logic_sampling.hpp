//
// Created by elle on 22/07/21.
//

#ifndef BAYLIB_LOGIC_SAMPLING_HPP
#define BAYLIB_LOGIC_SAMPLING_HPP

#define CL_TARGET_OPENCL_VERSION 220


#include <vector>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

#include <baylib/inference/abstract_inference_algorithm.hpp>

namespace bn {
    namespace inference{
        namespace compute = boost::compute;
        using boost::compute::lambda::_1;
        using boost::compute::lambda::_2;


        /** ===== Logic Sampling Algorithm ===
        *
        * This class represents the Logic Sampling approximate
        * inference algorithm for discrete Bayesian Networks.
        * The implementation uses boost::compute to exploit GPGPU optimization.
        * All samples are simulated in parallel, for this reason the maximum memory usage
        * tolerable must be specified to avoid filling up the memory of the device in case of large
        * number of samples.
        * @tparam Probability : the type expressing the probability
        **/
        template <typename Probability>
        class logic_sampling : public vectorized_inference_algorithm<Probability>{
            using prob_v = boost::compute::vector<Probability>;
        public:

            logic_sampling(
                    ulong samples,
                    size_t memory,
                    uint seed = 0,
                    const compute::device &device = compute::system::default_device()
            )
                    : vectorized_inference_algorithm<Probability>(samples, memory, seed, device)
            { }



            marginal_distribution<Probability> make_inference (const bn::bayesian_network<Probability> &bn){
                BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                          [](auto &var){ return bn::cpt_filled_out(var); }),
                              "conditional probability tables must be properly filled to"
                              " run logic_sampling inference algorithm",
                              std::runtime_error);

                auto [iter_samples, niter] = this->calculate_iterations(bn);
                auto vertex_queue = bn::sampling_order(bn);
                marginal_distribution<Probability> marginal_result(bn.begin(), bn.end());
                for (ulong i = 0; i< niter; i++) {

                    std::vector<bcvec> result_container(vertex_queue.size());
                    marginal_distribution<Probability> temp(bn.begin(), bn.end());

                    for(ulong v : vertex_queue) {

                        std::vector<bcvec> parents_result;

                        // Build parents result vector in the correct order
                        auto parents = bn[v].parents_info.names();
                        std::reverse(parents.begin(), parents.end());
                        for (auto p : parents) {
                            parents_result.push_back(result_container[bn.index_of(p)]);
                        }

                        auto result = this->simulate_node(bn[v].table(), parents_result, iter_samples);

                        // Save result in the data structure with the correct expiration date
                        result_container[v] = bcvec(result, bn[v].states().size(), bn.children_of(v).size());
                        auto accumulated_result = compute_result_general(result_container[v]);

                        for(int ix=0; ix< accumulated_result.size(); ix++)
                            marginal_result[v][ix] += accumulated_result[ix];

                        for(auto p: parents){
                            result_container[bn.index_of(p)].set_use();
                        }
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
            std::vector<ulong> compute_result_general(bcvec& res)
            {
                std::vector<ulong> acc_res(res.cardinality);
                for (bn::state_t i = 0; i < res.cardinality; ++i) {
                    acc_res[i] = compute::count(res.get_states().begin(), res.get_states().end(), i, this->queue);
                }
                return acc_res;
            }
        };
    } // namespace inference
} // namespace bn


#endif //BAYLIB_LOGIC_SAMPLING_HPP