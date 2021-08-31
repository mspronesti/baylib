//
// Created by paolo on 31/08/21.
//

#include <iostream>
#include <baylib/parser/xdsl_parser.hpp>

int main(){
    // To load a network from file use the xdsl_parser class
    bn::xdsl_parser<double> parser;
    auto network = parser.deserialize("../../examples/xdsl/Coma.xdsl");

    for (auto &var: network.variables()) {
        std::cout << var.name() << '\n';
        std::cout << var.table() << '\n';
    }

}
