#ifndef BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP
#define BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP

#include <baylib/network/bayesian_utils.hpp>
#include <baylib/probability/marginal_distribution.hpp>

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
    }


#endif //BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP