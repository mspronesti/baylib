#ifndef BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP
#define BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP



#include <baylib/network/bayesian_utils.hpp>
#include <baylib/probability/marginal_distribution.hpp>
#include <baylib/tools/random/random_generator.hpp>
#include <future>
#include <baylib/baylib_concepts.hpp>

//! \file abstract_inference_algorithm.hpp
//! \brief Abstract classes for stocastic algorithms

namespace baylib {
    namespace inference {
        /**
        * This class models a generic approximate
        * inference algorithm for discrete Bayesian
        * networks
        * @tparam Network_ the type of bayesian network
        */
        template <BNetDerived Network_>
        class inference_algorithm {
        public:
            typedef Network_ network_type;
            typedef typename network_type::variable_type variable_type;
            typedef typename variable_type::probability_type probability_type;

            /**
             * The abstract inference algorithm
             * only receives hyperparameters in the
             * constructor
             * @param nsamples : number of samples
             * @param nthreads : number of threads (default: 1)
             * @param seed     : custom seed for the generator (default: 0)
             */
            explicit inference_algorithm(
                    const network_type & bn,
                    unsigned long nsamples,
                    unsigned int seed = 0
            )
            : bn(bn)
            , nsamples(nsamples)
            , seed(seed)
            {}

            virtual ~inference_algorithm() = default;

            /**
             * Main method of the inference algorithm. Receives the bayesian network
             * but doesn't store it anywhere. It only uses its facilities to perform
             * inference thus avoiding unwanted copies and allowing reusability of the
             * same algorithm object
             * @param bn  : bayesian network
             * @return    : the marginal distributions
             */
            virtual baylib::marginal_distribution<probability_type> make_inference () = 0;

            void set_number_of_samples(unsigned long _nsamples) { nsamples = _nsamples; }

            void set_seed(unsigned int _seed) { seed = _seed; }

            const network_type & bn;
            unsigned long nsamples;
        protected:
            unsigned int seed;
        };

        /**
         * This class models an approximate inference algorithm
         * parallelized with C++ threads.
         * Its make_inference employs the well-known approach of splitting
         * the sampling work over the number of threads and merging the results
         * @tparam Network_  : the type of bayesian network
         */
        template < BNetDerived Network_ >
        class parallel_inference_algorithm : public inference_algorithm<Network_>
        {
        public:
            typedef Network_ network_type;
            using typename inference_algorithm<Network_>::probability_type;
            using  inference_algorithm<Network_>::bn;

            explicit parallel_inference_algorithm(
                    const network_type & bn,
                    unsigned long nsamples,
                    unsigned int nthreads = 1,
                    unsigned int seed = 0
            )
            : inference_algorithm<Network_>(bn, nsamples, seed)
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
            baylib::marginal_distribution<probability_type> make_inference() override
            {
                typedef std::future<baylib::marginal_distribution<probability_type>> result;
                BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                          [this](auto &var) { return baylib::cpt_filled_out(bn, var.id()); }),
                              "conditional probability tables must be properly filled to"
                              " run an inference algorithm",
                              std::runtime_error)

                baylib::marginal_distribution<probability_type> inference_result(bn.begin(), bn.end());
                std::vector<result> results;
                baylib::seed_factory sf(nthreads, this->seed);

                auto job = [this](ulong samples_per_thread, uint seed) {
                    return sample_step(samples_per_thread, seed);
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

            void set_number_of_threads(unsigned int _nthreads) {
                nthreads = _nthreads >= std::thread::hardware_concurrency() ?
                           std::thread::hardware_concurrency() : _nthreads > 0 ?
                                                                 _nthreads : 1;
            }

        protected:
            virtual baylib::marginal_distribution<probability_type> sample_step (
                    unsigned long nsamples_per_step,
                    unsigned int seed
            ) = 0;

            unsigned int nthreads;
        };

    } // namespace inference
} // namespace baylib

#endif //BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP

