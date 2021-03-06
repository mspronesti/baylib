//
// Created by elle on 07/09/21.
//

#ifndef BAYLIB_REJECTION_SAMPLING_HPP
#define BAYLIB_REJECTION_SAMPLING_HPP

#include <baylib/inference/abstract_inference_algorithm.hpp>
#include <baylib/tools/random/random_generator.hpp>

//! \file rejection_sampling.hpp
//! \brief Rejection Sampling implementation with multi-thread support

namespace baylib{
    namespace inference {
        /**
         *  ========== Rejection Sampling Algorithm =========
         * This class represents the rejection sampling approximate
         * inference algorithm for discrete bayesian networks.
         * It allows to specify a custom generator and a custom initial
         * seed.
         * as they're required by std::discrete_distribution
         * @tparam Network_ : the type of bayesian network (must inherit from baylib::bayesian_net)
         * @tparam Generator_ : the type of random generator
         *                  (default Mersenne Twister pseudo-random generator)
         *                  must overload min(), max() and the functional operator
         *                  to comply with std library constraints
         */
        template <
                BNetDerived Network_,
                STDEngineCompatible Generator_ = std::mt19937
                >
        class rejection_sampling : public parallel_inference_algorithm<Network_>
        {
            typedef Network_ network_type;
            using typename parallel_inference_algorithm<Network_>::probability_type;
            using parallel_inference_algorithm<Network_>::bn;
        public:
            explicit rejection_sampling(
                    const network_type & bn,
                    ulong nsamples,
                    uint nthreads = 1,
                    uint seed = 0
            )
            : parallel_inference_algorithm<Network_>(bn, nsamples, nthreads, seed)
            { };


        private:
            baylib::marginal_distribution<probability_type> sample_step (
                ulong nsamples,
                uint seed
            ) override
            {
                std::vector<baylib::state_t> var_state_values;
                Generator_ rnd_gen(seed);

                marginal_distribution<probability_type> marginal_distr(bn.begin(), bn.end());
                ulong nvars =  bn.number_of_variables();

                var_state_values = std::vector<baylib::state_t>(nvars, 0);
                std::vector<baylib::state_t> samples;
                for(ulong i = 0; i < nsamples; ++i) {
                    bool reject = false;
                    for (ulong n = 0; n < nvars; ++n) {
                        baylib::state_t state_val = prior_sample(n, var_state_values, rnd_gen);
                        // if evidences are not sampled accordingly, discard the samples
                        if(bn[n].is_evidence() && bn[n].evidence_state() != state_val)
                        {
                            reject = true;
                            break;
                        }

                        var_state_values[n] = state_val;
                        samples.push_back(state_val);
                    }

                    if(!reject) {
                        ulong vid = 0;
                        for(ulong s : samples)
                            ++marginal_distr[vid++][s];
                    }

                    samples.clear();
                }

                return marginal_distr;
            }

            baylib::state_t prior_sample(
                    unsigned long n,
                    const std::vector<baylib::state_t> &var_state_values,
                    Generator_ &gen
            )
            {
                baylib::condition c;
                // builds a condition using parents and
                // their states
                for(auto & p : bn.parents_of(n))
                    c.add(
                        p,
                        var_state_values[p]
                    );

                const auto& cpt = bn[n].table();
                // build discrete distribution from the parent states
                std::discrete_distribution<baylib::state_t> distr(cpt[c].begin(), cpt[c].end());
                return distr(gen);
            }
        };
    } // namespace inference
} //namespace baylib

#endif //BAYLIB_REJECTION_SAMPLING_HPP
