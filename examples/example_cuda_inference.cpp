
//
// Created by paolo on 28/10/21.
//

#include <baylib/inference/cuda/likelihood_weighting_cuda.hpp>
#include <baylib/smile_utils/smile_utils.hpp>

#include <iostream>
/**
 * Baylib implements several algorithms exploiting gpgpu parallelization.
 */

int main(int argc, char** argv){

    using namespace baylib;
    using namespace baylib::inference;

    baylib::xdsl_parser parser;
    // We use the Hailfinder network for this example
    auto bn = parser.deserialize("../../examples/xdsl/Hailfinder2.5.xdsl");

    // Gpu algorithms use montecarlo simulations to approximate inference results, all simulations are made
    // simultaneously, for this reason we have to take into account memory usage
    // For all gpu algorithms the first attribute will be the network, the second one will be the number of samples
    // to be generated and the third one will be the amount of memory on the opencl device available
    likelihood_weighting_cuda ls(bn, 10000);

    // Gpu algorithms offer the same external interface as all other baylib algorithms
    auto result = ls.make_inference();

    // The main advantage to using this kind of parallelization is that for high number of samples the
    // computation time raises very slowly in respect to classical algorithms (as long as enough memory is provided)
    std::cout << result << '\n';

}