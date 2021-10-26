//
// Created by paolo on 31/08/21.
//

#include <baylib/network/bayesian_net.hpp>
#include <baylib/inference/rejection_sampling.hpp>
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
    // B    C
    // \   /
    //  \ /
    //   v
    //   A
    //   |
    //   |
    //   v
    //   D

    // Use add_variable to add a new random variable with
    // its random states (default 2 states)
    ulong A = bn.add_variable(2); // 2 can be omitted
    ulong B = bn.add_variable();
    ulong C = bn.add_variable();
    ulong D = bn.add_variable(4); // D has 4 states

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
    uint i = 0;
    do {
        bn.set_variable_probability(A, 0, factory.get(), pb_vector[i++]);
        bn.set_variable_probability(A, 1, factory.get(), pb_vector[i++]);
    } while(factory.has_next());

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

    // we can now perform inference on the network
    // let's use the rejection sampling approximate inference algorithm
    // with 1e5 samples and 10 threads.
    // The usage of other algorithms is show in example_inference
    // with named_bayesian networks
    using namespace baylib::inference;
    rejection_sampling rs(bn, 10000, 10);

    // the make_inference() method retrieves a marginal distribution
    // for the entire network
    auto inf_result = rs.make_inference();
    // we can now print it as it is using the operator <<
    // or prettify it a little
    std::cout << "P(A=0) = " << inf_result[A][0] << '\n';
    std::cout << "P(A=1) = " << inf_result[A][1] << '\n';
    std::cout << "P(B=0) = " << inf_result[B][0] << '\n';
    std::cout << "P(B=1) = " << inf_result[B][1] << '\n';
    std::cout << "P(C=0) = " << inf_result[C][0] << '\n';
    std::cout << "P(C=1) = " << inf_result[C][1] << '\n';
    std::cout << "P(D=0) = " << inf_result[D][0] << '\n';
    std::cout << "P(D=1) = " << inf_result[D][1] << '\n';
    std::cout << "P(D=2) = " << inf_result[D][2] << '\n';
    std::cout << "P(D=3) = " << inf_result[D][3] << '\n';
    // to see how to add an evidence and make inference
    // using that constraint, have a look at example_inference.cpp
}
