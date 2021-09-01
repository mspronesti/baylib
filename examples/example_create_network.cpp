//
// Created by paolo on 31/08/21.
//

#include <baylib/network/bayesian_network.hpp>
#include <iostream>

int main(){

    // Network creation can be done manually

    bn::bayesian_network<double> bn;

    //
    // b    c
    // \   /
    //  \ /
    //   v
    //   a
    //   |
    //   |
    //   v
    //   d

    // Use add_variable to add a new random variable with its random states
    bn.add_variable("a", {"T", "F"});
    bn.add_variable("b", {"T", "F"});
    bn.add_variable("c", {"T", "F"});
    bn.add_variable("d", {"T", "F"});

    // Use add_dependency to add an edge between two variables
    bn.add_dependency("b", "a");
    bn.add_dependency("c", "a");
    bn.add_dependency("a", "d");

    // Condition object is used to specify elements in the cpt
    bn::condition c;

    bn.set_variable_probability("b", 0, c, .5);
    bn.set_variable_probability("b", 1, c, .5);

    bn.set_variable_probability("c", 0, c, .002);
    bn.set_variable_probability("c", 1, c, 1 - .002);

    // An alternative to building the condition by hand is using the condition factory util to generate
    // all the possible conditions of a cpt
    std::vector<float> pb_vector = {0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.5, 0.5};
    auto factory = bn::condition_factory<double>(bn["a"]);
    uint i=0;
    do{
        bn.set_variable_probability("a", 0, factory.get(), pb_vector[i++]);
        bn.set_variable_probability("a", 1, factory.get(), pb_vector[i++]);
    }while(factory.has_next());

    c.add("a", 0);
    bn.set_variable_probability("d", 0, c, .5);
    bn.set_variable_probability("d", 1, c, .5);


    c.add("a", 0);
    bn.set_variable_probability("d", 0, c, .9);
    bn.set_variable_probability("d", 1, c, .1);

    c.add("a", 1);
    bn.set_variable_probability("d", 0, c, .9);
    bn.set_variable_probability("d", 1, c, .1);

    for (auto &var: bn.variables()) {
        std::cout << var.name() << '\n';
        std::cout << var.table() << '\n';
    }
}
