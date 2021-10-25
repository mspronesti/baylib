//
// Created by paolo on 31/08/21.
//

#include <baylib/network/bayesian_net.hpp>
#include <iostream>

/*
 * The Bayesian Network Class is the main container of the baylib library.
 * This data structure holds both the graph structure and the CPT of each variable inside the network.
 * CPT memory usage is optimized with the integration of the Copy On Write paradigm (COW),
 * each time a new CPT is added if a duplicate is found the new one is discarded and a reference to the original
 * one is saved instead.
*/

int main(){
    using namespace baylib;
    bayesian_net<random_variable<double>> bn;
    // We want to create manually the following
    // Bayesain network
    // b    c
    // \   /
    //  \ /
    //   v
    //   a
    //   |
    //   |
    //   v
    //   d

    // Use add_variable to add a new random variable with
    // its random states (default 2 states)
    ulong A = bn.add_variable();
    ulong B = bn.add_variable();
    ulong C = bn.add_variable();
    ulong D = bn.add_variable();

    // Use add_dependency to add an edge between two variables
    bn.add_dependency(B, A);
    bn.add_dependency(C, A);
    bn.add_dependency(A, D);

    // Condition object is used to specify elements in the cpt
    baylib::condition c;

    bn.set_variable_probability(B, 0, c, .5);
    bn.set_variable_probability(B, 1, c, .5);

    bn.set_variable_probability(C, 0, c, .002);
    bn.set_variable_probability(C, 1, c, 1 - .002);

    // An alternative to building the condition by hand is using the condition factory util to generate
    // all the possible conditions of a cpt
    std::vector<float> pb_vector = {0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.5, 0.5};
    auto factory = baylib::condition_factory(bn, A);
    uint i=0;
    do{
        bn.set_variable_probability(A, 0, factory.get(), pb_vector[i++]);
        bn.set_variable_probability(A, 1, factory.get(), pb_vector[i++]);
    }while(factory.has_next());

    c.add(A, 0);
    bn.set_variable_probability(D, 0, c, .5);
    bn.set_variable_probability(D, 1, c, .5);


    c.add(A, 0);
    bn.set_variable_probability(D, 0, c, .9);
    bn.set_variable_probability(D, 1, c, .1);

    c.add(A, 1);
    bn.set_variable_probability(D, 0, c, .9);
    bn.set_variable_probability(D, 1, c, .1);

    // Bayesian network can be iterated over with a for loop
    for (auto &var: bn) {
        std::cout << var.id() << '\n';
        std::cout << var.table() << '\n';
    }
}
