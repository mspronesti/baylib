#ifndef BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP
#define BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP

#include <baylib/network/bayesian_utils.hpp>
#include <baylib/probability/marginal_distribution.hpp>
#include <baylib/tools/random/random_generator.hpp>

#include <future>

namespace bn {
    namespace inference {
        /**
        * This class models a generic approximate
        * inference algorithm for discrete Bayesian
        * networks
        * @tparam Probability
        */
        template<typename Probability>
        class inference_algorithm {
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
                    unsigned int seed = 0
            )
                    : nsamples(nsamples), seed(seed) {}

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
                    const bn::bayesian_network<Probability> &bn
            ) = 0;

            void set_number_of_samples(unsigned long _nsamples) { nsamples = _nsamples; }

            void set_seed(unsigned int _seed) { seed = _seed; }

        protected:
            unsigned long nsamples;
            unsigned int seed;
        };

        /**
         * This class models an approximate inference algorithm
         * parallelized with C++ threads.
         * Its make_inference employs the well-known approach of splitting
         * the sampling work over the number of threads and merging the results
         * @tparam Probability  : the type expressing probability
         */
        template<typename Probability>
        class parallel_inference_algorithm : public inference_algorithm<Probability> {
        public:
            explicit parallel_inference_algorithm(
                    unsigned long nsamples,
                    unsigned int nthreads = 1,
                    unsigned int seed = 0
            )
            : inference_algorithm<Probability>(nsamples, seed)
            {
                set_number_of_threads(nthreads);
            }

            /**
             * Models the standard approach towards MCMC parallelization,
             * i.e. assigns the sampling step to the number of available threads
             * and eventually merges the results
             * @param bn : bayesian network graph
             * @return   : the marginal distribution of the variables post inference
             */
            bn::marginal_distribution<Probability> make_inference(
                    const bn::bayesian_network<Probability> &bn
            ) override
            {
                typedef std::future<bn::marginal_distribution<Probability>> result;
                BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                          [](auto &var) { return bn::cpt_filled_out(var); }),
                              "conditional probability tables must be properly filled to"
                              " run gibbs sampling inference algorithm",
                              std::runtime_error)

                bn::marginal_distribution<Probability> inference_result(bn.begin(), bn.end());
                std::vector<result> results;
                bn::seed_factory sf(nthreads, this->seed);

                auto job = [this, &bn](ulong samples_per_thread, uint seed) {
                    return sample_step(bn, samples_per_thread, seed);
                };

                ulong samples_per_thread = this->nsamples / nthreads;
                // assigning jobs
                for (uint i = 0; i < nthreads - 1; ++i)
                    results.emplace_back(std::async(job, samples_per_thread, sf.get_new()));

                // last thread (doing the extra samples if nsamples % nthreads != 0)
                ulong left_samples = this->nsamples - (nthreads - 1) * samples_per_thread;
                results.emplace_back(std::async(job, samples_per_thread, sf.get_new()));

                // accumulate results of each parallel execution
                for (auto &res: results)
                    inference_result += res.get();

                // normalize the distribution before retrieving it
                inference_result.normalize();
                return inference_result;
            }

            void set_number_of_threads(unsigned int _nthreads)
            {
                nthreads = _nthreads >= std::thread::hardware_concurrency() ?
                           std::thread::hardware_concurrency() : _nthreads > 0 ?
                                                                 _nthreads : 1;
            }

        protected:
            virtual bn::marginal_distribution<Probability> sample_step(
                    const bn::bayesian_network<Probability> &bn,
                    unsigned long nsamples_per_step,
                    unsigned int seed
            ) = 0;

            unsigned int nthreads;
        };

        /**
         * This class models an approximate inference algorithm
         * vectorized with a GPGPU approach.
         * <further details here @paolo>
         * @tparam Probability  : the type expressing probability
         */
        template <typename Probability>
        class vectorized_inference_algorithm /*: public inference_algorithm<Probability>*/ {
         // TODO: to be implemented (@paolo)
        public:
        protected:
            size_t memory;
        };

    } // namespace inference
} // namespace bn

#endif //BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP