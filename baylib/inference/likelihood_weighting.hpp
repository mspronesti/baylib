//
// Created by elle on 30/08/21.
//

#ifndef BAYLIB_LIKELIHOOD_WEIGHTING_HPP
#define BAYLIB_LIKELIHOOD_WEIGHTING_HPP

#include <random>
#include <future>
#include <baylib/tools/random/random_generator.hpp>
#include <baylib/inference/abstract_inference_algorithm.hpp>

//! \file likelihood_weighting.hpp
//! \brief Likelihood weighting implementation with multi-thread support

namespace bn {
    namespace  inference {
        /** ===== Likelihood Weighting Algorithm ===
         *
         * This class represents the likelihood-weighting approximate
         * inference algorithm for discrete Bayesian Networks.
         * It offers the possibility to use a custom generator and
         * an initial seed
         * @tparam Probability : the type expressing the probability
         * @tparam Generator_  : the random generator
         *                     (default Mersenne Twister pseudo-random generator)
         */
        template <
                BNetDerived Network_,
                typename Generator_ = std::mt19937
                >
        class likelihood_weighting : public parallel_inference_algorithm<Network_>
        {
            typedef Network_ network_type;
            using typename parallel_inference_algorithm<Network_>::probability_type;
            using parallel_inference_algorithm<Network_>::bn;
            typedef std::vector<ulong> pattern_t;
        public:
            explicit likelihood_weighting(
                    const network_type & bn,
                    ulong nsamples,
                    uint nthreads = 1,
                    uint seed = 0
            )
            : parallel_inference_algorithm<Network_>(bn, nsamples, nthreads, seed)
            { };
        private:
            bn::marginal_distribution<probability_type> sample_step(
                 ulong nsamples,
                 uint seed
            )
            {
                bn::marginal_distribution<probability_type> mdistr(bn.begin(), bn.end());
                bn::random_generator<probability_type, Generator_> rnd_gen(seed);

                for(ulong i=0; i<nsamples; i++){
                    auto sample_pair = weighted_sample(rnd_gen);
                    ulong vid = 0;
                    auto weight = sample_pair.second;

                    for(auto & samp : sample_pair.first)
                        mdistr[vid++][samp] += weight ;
                }
                return mdistr;
            }

            std::pair<pattern_t , probability_type> weighted_sample(
                bn::random_generator<probability_type> & rnd_gen
            )
            {

                probability_type weight = 1.0;
                auto pattern = pattern_t(bn.number_of_variables(), 0.0);

                for(ulong vid = 0; vid < bn.number_of_variables(); ++vid)
                {
                    auto & var = bn[vid];
                    bn::condition parent_state;

                    for(auto par : bn.parents_of(vid))
                        parent_state.add(
                                par,
                                pattern[par]
                        );

                    const auto & cpt = var.table();
                    if(var.is_evidence()) {
                        ulong evidence_state = var.evidence_state();
                        weight *= cpt[parent_state][evidence_state];
                        pattern[vid] = evidence_state;
                    } else {
                        pattern[vid] = make_random_by_weight(
                                rnd_gen.get_random(),
                                cpt[parent_state]
                        );
                    }
                }

                return std::make_pair(pattern, weight);
            }

            uint make_random_by_weight(
                    const probability_type p,
                    const std::vector<probability_type> & weight
            )
            {
                BAYLIB_ASSERT(0.0 <= p && p <= 1.0,
                              "Invalid probability value"
                              " not included in [0,1]",
                              std::logic_error);

                probability_type total = 0.0;
                for(uint i = 0; i < weight.size(); ++i)
                {
                    auto const old_total = total;
                    total += weight[i];
                    if(old_total <= p && p < total)
                    {
                        return i;
                    }
                }

                return weight.size() - 1;
            }

        };
    } // namespace inference
} // namespace bn

#endif //BAYLIB_LIKELIHOOD_WEIGHTING_HPP
