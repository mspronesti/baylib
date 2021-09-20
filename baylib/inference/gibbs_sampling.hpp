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
        template <typename Probability, typename Generator=std::mt19937>
        class gibbs_worker{
            /**
             * This class represents a single worker
             * i.e. a thread performing a complete simulation
             * of the bayesian network graph nsamples time.
             * Each Gibbs sampler is responsible of n_total_samples/n_workers
             * simulations
             */
        public:
            gibbs_worker(
                    const bn::bayesian_network<Probability> &bn,
                    ulong nsamples,
                    uint seed = 0
            )
            : nsamples(nsamples)
            , bn(bn)
            , var_state_values(std::vector<bn::state_t>(bn.number_of_variables(), 0))
            , rnd_gen(seed)
            { }

            bn::marginal_distribution<Probability> sample(){
                bn::marginal_distribution<Probability> marginal_distr(bn.begin(), bn.end());
                ulong nvars = bn.number_of_variables();

                for(ulong i = 0; i < nsamples; ++i)
                    for(ulong n = 0; n < nvars; ++n)
                    {
                        auto& var = bn[n] ;
                        auto res = sample_single_variable(var);
                        ++marginal_distr[res.first][res.second];
                    }

                return marginal_distr;
            }

        private:
            // contains, for each variable, the current state value
            std::vector<bn::state_t> var_state_values;

            const bn::bayesian_network<Probability> &bn;
            bn::random_generator<Probability, Generator> rnd_gen;
            ulong nsamples;
            
            std::pair<ulong, ulong> sample_single_variable( bn::random_variable<Probability> & var )
            {
                ulong index = var.id();
                if(var.is_evidence()) {
                    var_state_values[index] = var.evidence_state();
                    return {index, var.evidence_state()};
                }

                auto samples = std::vector<Probability>(var.states().size(), 0.0);
                for(ulong i = 0; i < samples.size(); ++i){
                    var_state_values[index] = i;
                    // here we evaluate P(Xi | x_t, t = 1, 2, ..., i-1, 1+1, ..., n)
                    // which is P(Xi | markov_blanket(Xi))
                    // which is proportional to
                    //  P(Xi | parents(Xi)) * prod_{j=1}^{k} P(Yj | parents(Yj))
                    //
                    // where
                    // - prod is the product from j = 1 to k
                    // - k is the number of children of Xi
                    // - Yj is the j-th child of X
                    samples[i] = get_probability(index);
                    for(ulong j : bn.children_of(index))
                        samples[i] *= get_probability(j);
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
                var_state_values[index] = j;
                return {index, j};
            }

            /**
             * Get the probability of the current realization of a specific node
             * @param n : numerical identifier of node
             * @return  : Probability of the current realization of n
             */
            Probability get_probability ( const unsigned long n )
            {
                bn::condition c;
                // builds a condition using parents and
                // their states
                for(auto & p : bn.parents_of(n))
                    c.add(
                       bn[p].name(),
                       var_state_values[p]
                    );

                const auto& cpt = bn[n].table();
                return  cpt[c][var_state_values[n]];
            }
        };


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
        class gibbs_sampling : public parallel_inference_algorithm<Probability>{
        public:
            explicit gibbs_sampling(
                    ulong nsamples,
                    uint nthreads = 1,
                    uint seed = 0
            )
            : parallel_inference_algorithm<Probability>(nsamples, nthreads, seed)
            { }

        private:
            bn::marginal_distribution<Probability> sample_step(
                    const bn::bayesian_network<Probability> & bn,
                    unsigned long nsamples,
                    unsigned int seed
            ) override
            {
                auto worker = gibbs_worker(bn, nsamples, seed);
                return worker.sample();
            }
        };

    }  // namespace inference
} // namespace bn

#endif //BAYLIB_GIBBS_SAMPLING_HPP