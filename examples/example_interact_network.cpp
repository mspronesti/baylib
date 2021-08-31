//
// Created by paolo on 31/08/21.
//

#include <baylib/parser/xdsl_parser.hpp>
#include <iostream>

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
    for (auto &probi: var.table()) {
        for(auto &probj: probi)
            std::cout << probj << ' ';
        std::cout << '\n';
    }

}