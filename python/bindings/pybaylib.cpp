//
// Created by elle on 02/11/21.
//
#include "inference.h"
#include "bayesian.h"

PYBIND11_MODULE(_pybaylib, m) {
    bind_conditional_probability_table(m);
    bind_random_variable(m);
    bind_bayesian_network(m);
    bind_marginal_distribution(m);

    // inference algorithms
    bind_parallel_inference_algorithm<gibbs_sampl>(m, "gibbs_sampling");
}
