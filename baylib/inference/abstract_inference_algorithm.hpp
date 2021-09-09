#ifndef BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP
#define BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP

#include <baylib/network/bayesian_utils.hpp>
#include <baylib/probability/marginal_distribution.hpp>
#include <baylib/tools/random/random_generator.hpp>

#include <future>

namespace bn::inference {
        /**
        * This class models a generic approximate
        * inference algorithm for discrete Bayesian
        * networks
        * @tparam Probability
        */
        template <typename Probability>
        class inference_algorithm  {
        public:
            /**
             * The abstract inference algorithm
             * only receives hyperparameters in the
             * constructor
             * @param nsamples : number of samples
             * @param nthreads : number of threads (default: 1)
             * @param seed     : custom seed for the generator (default: 0)
             */
            explicit inference_algorithm(
                    unsigned long nsamples,
                    unsigned int nthreads = 1,
                    unsigned int seed = 0
            )
            : nsamples(nsamples)
            , nthreads(nthreads)
            , seed(seed)
             { }

             virtual ~inference_algorithm() = default;

             /**
              * Main method of the inference algorithm. Receives the bayesian network
              * but doesn't store it anywhere. It only uses its facilities to perform
              * inference thus avoiding unwanted copies and allowing reusability of the
              * same algorithm object
              * @param bn  : bayesian network
              * @return    : the marginal distributions
              */
            virtual bn::marginal_distribution<Probability> make_inference(
                    const bn::bayesian_network<Probability> & bn
            ) = 0;

        protected:
            unsigned long nsamples;
            unsigned int nthreads;
            unsigned int seed;
        };

        /**
         * Utility to split sampling jobs to given number of threads
         * and eventually sum their results.
         * Used by approximate inference algorithms.
         * @tparam Probability  : the type expressing the probability
         * @tparam F            : function type
         * @param bn            : bayesian network
         * @param nsamples      : total number of samples
         * @param nthreads      : total number of threads
         * @param seed          : initial seed
         * @param job           : job to perform for each thread
         * @return              : final marginal distribution (inference result)
         */
        template <typename Probability, typename F>
        bn::marginal_distribution<Probability> assign_and_compute(
            const bn::bayesian_network<Probability> & bn,
            const F & job,
            ulong nsamples,
            uint nthreads,
            uint seed
        )
        {
            typedef std::future<bn::marginal_distribution<Probability>> result;
            BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                   [](auto &var){ return bn::cpt_filled_out(var); }),
                       "conditional probability tables must be properly filled to"
                       " run gibbs sampling inference algorithm",
                       std::runtime_error)

            bn::marginal_distribution<Probability> inference_result(bn.begin(), bn.end());
            std::vector<result> results;
            bn::seed_factory sf(nthreads, seed);

            ulong samples_per_thread = nsamples / nthreads;

            // assigning jobs
            for(uint i = 0; i < nthreads - 1; ++i)
                results.emplace_back(std::async(job, samples_per_thread, sf.get_new() ));

            // last thread (doing the extra samples if nsamples % nthreads != 0)
            ulong left_samples = nsamples - (nthreads - 1) * samples_per_thread;
            results.emplace_back(std::async(job, samples_per_thread, sf.get_new() ));

            // accumulate results of each parallel execution
            for(auto & res: results)
                inference_result += res.get();

            // normalize the distribution before retrieving it
            inference_result.normalize();
            return inference_result;
        }
    }


#endif //BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP