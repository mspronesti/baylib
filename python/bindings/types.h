//
// Created by elle on 02/11/21.
//
#ifndef BAYLIB_TYPES_H
#define BAYLIB_TYPES_H

#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>
#include <baylib/inference/rejection_sampling.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/inference/adaptive_importance_sampling.hpp>
#include <baylib/smile_utils/smile_utils.hpp>

using namespace baylib;
using namespace baylib::cow;
using namespace baylib::inference;

// baylib types
typedef double probability_;
typedef random_variable<probability_> random_var;
typedef bayesian_net<random_var> bayesian_network;
typedef marginal_distribution<probability_> marginal_distr;
typedef condition_factory<bayesian_network> condition_factory;
typedef cpt<probability_> _cpt;

// algorithms
typedef inference_algorithm<bayesian_network> abstract_inference_algorithm;
typedef parallel_inference_algorithm<bayesian_network> _par_inference_algorithm;
typedef vectorized_inference_algorithm<bayesian_network> _vec_inference_algorithm;
typedef gibbs_sampling<bayesian_network> gibbs_sampl;
typedef rejection_sampling<bayesian_network> rejection_sampl;
typedef likelihood_weighting<bayesian_network> likelihood_w;
typedef logic_sampling<bayesian_network> logic_sampl;
typedef adaptive_importance_sampling<bayesian_network> importance_sampl;

// smile
typedef named_random_variable<probability_> named_random_var;
typedef bayesian_net<named_random_var> named_bayesian_net;
typedef xdsl_parser<probability_> xdsl_parser;

#endif //BAYLIB_TYPES_H
