//
// Created by paolo on 31/08/21.
//

#include <baylib/parser/xdsl_parser.hpp>
#include <iostream>

/*
 * Model exploration is supported by enabling modifications after initialization of both the network
 * structure or the contents of the CPTs. CPTs contents are copied only when needed after a modification.
 * Networks obtained through copying manifest the same behavior regarding CPT optimization, the content of the CPT
 * will not be really copied until an explicit modification is made.
 */

int main(){
    bn::xdsl_parser<double> parser;
    auto network = parser.deserialize("../../examples/xdsl/Coma.xdsl");

    // Variables data can be accessed through the method variable or the [] operator

    bn::random_variable<double> var;
    var = network.variable(0);
    var = network[0];
    var = network.variable("Coma");
    var = network["Coma"];

    // random variable holds the data about names of the states and the cpt
    for (auto &state: var.states())
        std::cout << state << '\n';

    // Access is supported both by single element and iterators
    for (auto &prob_i: var.table()) {
        for(auto &prob_j: prob_i)
            std::cout << prob_j << ' ';
        std::cout << '\n';
    }

}