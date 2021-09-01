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
        class abstract_inference_algorithm {
        public:
            explicit abstract_inference_algorithm(
                    unsigned long nsamples,
                    unsigned int nthreads = 1,
                    unsigned int seed = 0
            )
            : nsamples(nsamples)
            , nthreads(nthreads)
            , seed(seed)
             { }

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