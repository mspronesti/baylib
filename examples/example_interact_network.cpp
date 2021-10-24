//
// Created by paolo on 31/08/21.
//

#include <baylib/smile_utils/smile_utils.hpp>
#include <iostream>

/*
 * Model exploration is supported by enabling modifications after initialization of both the network
 * structure or the contents of the CPTs. CPTs contents are copied only when needed after a modification.
 * Networks obtained through copying manifest the same behavior regarding CPT optimization, the content of the CPT
 * will not be really copied until an explicit modification is made.
 */

int main(){
    baylib::xdsl_parser<double> parser;
    auto network = parser.deserialize("../../examples/xdsl/Coma.xdsl");

    // Variables data can be accessed through the method variable or the [] operator

    baylib::named_random_variable<double> var;
    var = network.variable(0);
    var = network[0];

    // indexes can be recovered using the name_map provided with the make_name_map util

    auto name_map = baylib::make_name_map(network);
    var = network.variable(name_map["Coma"]);
    var = network[name_map["Coma"]];

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