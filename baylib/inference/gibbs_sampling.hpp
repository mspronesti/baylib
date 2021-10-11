//
// Created by elle on 18/08/21.
//

#ifndef BAYLIB_GIBBS_SAMPLING_HPP
#define BAYLIB_GIBBS_SAMPLING_HPP

#include <baylib/inference/abstract_inference_algorithm.hpp>
#include <baylib/tools/random/random_generator.hpp>

#include <algorithm>
#include <future>

//! \file gibbs_sampling.hpp
//! \brief Gibbs Sampling implementation with multi-thread support

namespace bn {
    namespace inference {
        /** ===== Gibbs sampling Algorithm ===
         *
         * This class represents the Gibbs Sampling approximate
         * inference algorithm for discrete Bayesian Networks.
         * It's based on the Gibbs sampler.
         * It offers the possibility to use a custom generator and
         * an initial seed
         * NOTICE: Gibbs sampling should not be used for bayesian networks
         *         with deterministic nodes, i.e. nodes with some entry
         *         in the cpt equal to 1.0
         * @tparam Probability : the type expressing the probability
         * @tparam Generator  : the random generator
         *                     (default Mersenne Twister pseudo-random generator)
         */
        template <typename Probability = double, typename Generator=std::mt19937>
        class gibbs_sampling : public parallel_inference_algorithm<Probability, gibbs_sampling<Probability, Generator>>{
        public:
            explicit gibbs_sampling (
                    ulong nsamples,
                    uint nthreads = 1,
                    uint seed = 0
            )
            : parallel_inference_algorithm<Probability, gibbs_sampling<Probability, Generator>>(nsamples, nthreads, seed)
            { }

            template<class Variable>
            bn::marginal_distribution<Probability> sample_step (
                    const bn::bayesian_network<Variable> & bn,
                    unsigned long nsamples, // the number of samples of each thread
                    unsigned int seed
            )
            {
                ulong nvars = bn.number_of_variables();
                // contains, for each variable, the current state value
                auto var_state_values = std::vector<bn::state_t>(nvars);

                bn::random_generator<Probability, Generator> rnd_gen(seed);
                bn::marginal_distribution<Probability> marginal_distr(bn.begin(), bn.end());

                for(ulong i = 0; i < nsamples; ++i)
                    for(ulong n = 0; n < nvars; ++n)
                    {
                        auto sample = sample_single_variable(bn, n, var_state_values, rnd_gen);
                        ++marginal_distr[n][sample];
                    }

                return marginal_distr;
            }

        private:
            /**
             * Samples a single variable usign the Gibbs sampling
             * algorithm
             * @param bn  : the bayesian network
             * @param n   : the index of the sampled variable
             * @param var_state_values : the vector containing the var states
             * @param rnd_gen  : the random generator with the given seed
             * @return sampled state
             */
            template<class Variable>
            ulong sample_single_variable(
                const bn::bayesian_network<Variable> &bn,
                const unsigned long n,
                std::vector<bn::state_t> &var_state_values,
                bn::random_generator<Probability, Generator> &rnd_gen
            )
            {
                auto var = bn[n];
                if(var.is_evidence()) {
                    var_state_values[n] = var.evidence_state();
                    return var.evidence_state();
                }

                auto samples = std::vector<Probability>(var.number_of_states(), 0.0);
                for(ulong i = 0; i < samples.size(); ++i) {
                    var_state_values[n] = i;
                    // here we evaluate P(Xi | x_t, t = 1, 2, ..., i-1, 1+1, ..., n)
                    // which is P(Xi | markov_blanket(Xi))
                    // which is proportional to
                    //  P(Xi | parents(Xi)) * prod_{j=1}^{k} P(Yj | parents(Yj))
                    //
                    // where
                    // - prod is the product from j = 1 to k
                    // - k is the number of children of Xi
                    // - Yj is the j-th child of X
                    samples[i] = get_probability(bn, n, var_state_values);
                    for(ulong j : bn.children_of(n))
                        samples[i] *= get_probability(bn, j, var_state_values);
                }
                // normalize
                Probability sum = std::accumulate(samples.begin(), samples.end(), 0.0);
                std::for_each(samples.begin(), samples.end(), [sum](auto & val){
                    val /= sum;
                });

                Probability prob = rnd_gen.get_random();
                ulong j;
                for(j = 0; j < samples.size() - 1; ++j)
                {
                    if(prob <= samples[j])
                        break;
                    else
                        prob -= samples[j];
                }
                var_state_values[n] = j;
                return j;
            }

            /**
            * Get the probability of the current realization of a specific node
            * @param n : numerical identifier of node
            * @return  : Probability of the current realization of n
            */
            template<class Variable>
            Probability get_probability (
                const bn::bayesian_network<Variable> &bn,
                const unsigned long n,
                const std::vector<bn::state_t> &var_state_values
            )
            {
                bn::condition c;
                // builds a condition using parents and
                // their states
                for(auto & p : bn.parents_of(n))
                    c.add(
                            p,
                            var_state_values[p]
                    );

                const auto& cpt = bn[n].table();
                return  cpt[c][var_state_values[n]];
            }
        };

    }  // namespace inference
} // namespace bn

#endif //BAYLIB_GIBBS_SAMPLING_HPP