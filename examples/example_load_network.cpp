//
// Created by paolo on 31/08/21.
//

#include <iostream>
#include <baylib/smile_utils/smile_utils.hpp>


/*
 * The Bayesian Network Class is the main container of the baylib library.
 * Creation of the network can be done through the xdsl_parser class,
 * xdsl format is a xml derived format.
 * More details on the format can be found here:
 * https://support.bayesfusion.com/docs/
*/

int main(){

    baylib::xdsl_parser<double> parser;
    auto network = parser.deserialize("../../examples/xdsl/Coma.xdsl");

    for (auto &var: network) {
        std::cout << var.name() << '\n';
        std::cout << var.table() << '\n';
    }

}
