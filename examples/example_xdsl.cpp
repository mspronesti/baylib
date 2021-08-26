//
// Created by elle on 25/08/21.
//
#include <iostream>

#include <baylib/parser/net_parser.hpp>
#include <baylib/inference/gibbs_sampling.hpp>


void test_gibbs(const std::string &filename, uint njobs = 1){
    std::cout << filename << '\n';
    bn::net_parser<double> parser;
    auto net = parser.load_from_xdsl("../../examples/xdsl/" + filename);

    auto gibbs = bn::inference::gibbs_sampling<double>{net};
    gibbs.inferenciate(10000, njobs);
    std::cout << gibbs.inference_result();
    for(int i = 0; i < net.number_of_variables(); ++i)
        std::cout << net[i].name() << '\n';
    std::cout << "\n\n\n";
}

int main(){
    test_gibbs("Coma.xdsl", 10 );
    test_gibbs("Credit.xdsl");
    test_gibbs("Animals.xdsl");
    test_gibbs("VentureBN.xdsl");
}