//
// Created by paolo on 31/08/21.
//

#include <baylib/parser/xdsl_parser.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>
#include <iostream>


/*
 * Inference algorithms are the main feature of the baylib library, they are all located under
 * the bn::inference namespace, and they all inherit from the abstract class inference_algorithm.
 * All algorithms have 2 methods:
 * - The constructor that takes in input the hyperparameters and settings of the algorithm
 * - The make_inference method that takes in input a reference of the bayesian network and returns a marginal_probability structure
 */

void example_logic_sampling(const bn::bayesian_network<double>& net){
    /*
     * Logic sampling is a simple sampling algorithm, the implementation uses openCL to make
     * the relevant computations on GPGPU if it's available; custom openCL devices can also be used
     * by passing them through the constructor.
     * The algorithm is nearly invariant to the number of samples requested as long as enough memory
     * on the device is available, for this reason in the constructor you must specify how much memory
     * the algorithm can use, if not enough memory is provided for the requested number of samples
     * the algorithm still works but can perform slower than expected.
    */
    bn::inference::logic_sampling<double> logic(2*std::pow(2,30), 100000);
    auto result = logic.make_inference(net);
    std::cout << "LOGIC SAMPLING:\n";
    std::cout << result;
}

void example_gibbs_sampling(const bn::bayesian_network<double>& net){
    /*
     * Gibbs sampling is a simple MCMC algorithm, the implementation is based on multithreading using
     * std::async from c++11, for low number of samples this algorithm or likelihood sampling is
     * preferable to logic_sampling
     */
    bn::inference::gibbs_sampling<double> gibbs(10000, 4);
    auto result = gibbs.make_inference(net);
    std::cout << "GIBBS SAMPLING:\n";
    std::cout << result;
}

void example_likelihood_weighing(const bn::bayesian_network<double>& net){
    /*
     * Likelihood sampling is a simple sampling algorithm, the implementation is based on multithreading using
     * std::async from c++11, for networks without evidence this algorithm is conceptually very similar to logic_sampling,
     * this can be used for making benchmarks on performance of multithreading approach vs GPGPU.
     */
    bn::inference::likelihood_weighting<double> likely(10000, 4);
    auto result = likely.make_inference(net);
    std::cout << "LIKELIHOOD WEIGHTING:\n";
    std::cout << result;
}

int main(){
    bn::xdsl_parser<double> parser;
    auto network = parser.deserialize("../../examples/xdsl/Hailfinder2.5.xdsl");
    example_logic_sampling(network),
    example_gibbs_sampling(network);
    example_likelihood_weighing(network);
}