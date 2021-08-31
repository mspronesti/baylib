//
// Created by paolo on 31/08/21.
//

#include <baylib/parser/xdsl_parser.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <iostream>


int main(){
    bn::xdsl_parser<double> parser;
    auto network = parser.deserialize("../../examples/xdsl/Coma.xdsl");

    // To make inference use any algorithm that inherits from abstract_inference_algorithm
    // Hyperparameters are set in the constructor
    bn::inference::gibbs_sampling<double> alg1(10000, 5);
    bn::inference::logic_sampling<double> alg2(2*std::pow(2,30), 100000);

    // The abstract method make_inference is used to call the algorithm selected and obtain the marginal_distribution
    // struct that holds the result
    auto result1 = alg1.make_inference(network);
    auto result2 = alg2.make_inference(network);

    std::cout << result1[0][0] << ' ' << result1[0][1] << '\n';
    std::cout << result2[0][0] << ' ' << result2[0][1] << '\n';
    std::cout << result1[network.index_of("Coma")][0] << ' ' << result1[network.index_of("Coma")][1] << '\n';
    std::cout << result2[network.index_of("Coma")][0] << ' ' << result2[network.index_of("Coma")][1] << '\n';
}