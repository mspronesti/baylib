//
// Created by elle on 30/08/21.
//

#ifndef BAYLIB_LIKELIHOOD_WEIGHTING_HPP
#define BAYLIB_LIKELIHOOD_WEIGHTING_HPP

#include <random>
#include <future>
#include <baylib/tools/random/random_generator.hpp>
#include <baylib/inference/abstract_inference_algorithm.hpp>

namespace bn {
    namespace  inference {
        /** ===== Likelihood Weighting Algorithm ===
         *
         * This class represents the likelihood-weighting approximate
         * inference algorithm for discrete Bayesian Networks.
         * It offers the possibility to use a custom generator and
         * an initial seed
         * @tparam Probability : the type expressing the probability
         * @tparam Generator  : the random generator
         *                     (default Mersenne Twister pseudo-random generator)
         */
        template<typename Probability = double, typename Generator = std::mt19937>
        class likelihood_weighting : public parallel_inference_algorithm<Probability>
        {
            typedef std::vector<ulong> pattern_t;
        public:
            explicit likelihood_weighting(
                    ulong nsamples,
                    uint nthreads = 1,
                    uint seed = 0
            )
            : parallel_inference_algorithm<Probability>(nsamples, nthreads, seed)
            { };

        private:
            bn::marginal_distribution<Probability> sample_step(
                 const bn::bayesian_network<Probability> &bn,
                 ulong nsamples,
                 uint seed
            ) override
            {
                bn::marginal_distribution<Probability> mdistr(bn.begin(), bn.end());
                bn::random_generator<Probability, Generator> rnd_gen(seed);

                for(ulong i=0; i<nsamples; i++){
                    auto sample_pair = weighted_sample(bn, rnd_gen);
                    ulong vid = 0;
                    auto weight = sample_pair.second;

                    for(auto & samp : sample_pair.first)
                        mdistr[vid++][samp] += weight ;
                }
                return mdistr;
            }


            std::pair<pattern_t , Probability> weighted_sample(
                const  bn::bayesian_network<Probability> &bn,
                bn::random_generator<Probability> & rnd_gen
            )
            {

                Probability weight = 1.0;
                auto pattern = pattern_t(bn.number_of_variables(), 0.0);

                for(ulong vid = 0; vid < bn.number_of_variables(); ++vid)
                {
                    auto & var = bn[vid];
                    bn::condition parent_state;

                    for(auto par : bn.parents_of(vid))
                        parent_state.add(
                                bn[par].name(),
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
                    const Probability p,
                    const std::vector<Probability> & weight
            )
            {
                BAYLIB_ASSERT(0.0 <= p && p <= 1.0,
                              "Invalid probability value"
                              " not included in [0,1]",
                              std::logic_error);

                Probability total = 0.0;
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
