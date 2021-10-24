//
// Created by paolo on 24/10/21.
//

//
// Created by paolo on 31/08/21.
//

#include <baylib/network/bayesian_net.hpp>
#include <baylib/smile_utils/smile_utils.hpp>
#include <iostream>

/*
 * The Bayesian Network Class is the main container of the baylib library, it can contain normal random
 * variables or any custom class as long as it inherits from the random_variable class, one of such examples
 * is the named_random_variable class offered in the smile_utils module,
 * where we offer on top of the normal features of random_variable
 * the possibility to give names to the variable and the conditions of the cpt table
*/

int main(){


    baylib::bayesian_net<baylib::named_random_variable<double>> bn;

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
    bn.add_variable("Gandalf", std::vector<std::string>{"T", "F"});
    bn.add_variable("Saruman", std::vector<std::string>{"T", "F"});
    bn.add_variable("Aragorn", std::vector<std::string>{"T", "F"});
    bn.add_variable("Sauron", std::vector<std::string>{"T", "F"});

    // We can retrieve the indexes of the variables starting from their names
    auto name_map = baylib::make_name_map(bn);

    // Use add_dependency to add an edge between two variables
    bn.add_dependency(name_map["Saruman"], name_map["Gandalf"]);
    bn.add_dependency(name_map["Aragorn"], name_map["Gandalf"]);
    bn.add_dependency(name_map["Gandalf"], name_map["Sauron"]);

    // Condition object is used to specify elements in the cpt
    baylib::condition c;

    bn.set_variable_probability(name_map["Saruman"], 0, c, .5);
    bn.set_variable_probability(name_map["Saruman"], 1, c, .5);

    bn.set_variable_probability(name_map["Aragorn"], 0, c, .002);
    bn.set_variable_probability(name_map["Aragorn"], 1, c, 1 - .002);

    // An alternative to building the condition by hand is using the condition factory util to generate
    // all the possible conditions of a cpt
    std::vector<float> pb_vector = {0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.5, 0.5};
    auto factory = baylib::condition_factory(bn, name_map["Gandalf"]);
    uint i=0;
    do{
        bn.set_variable_probability(name_map["Gandalf"], 0, factory.get(), pb_vector[i++]);
        bn.set_variable_probability(name_map["Gandalf"], 1, factory.get(), pb_vector[i++]);
    }while(factory.has_next());

    c.add(name_map["Gandalf"], 0);
    bn.set_variable_probability(name_map["Sauron"], 0, c, .5);
    bn.set_variable_probability(name_map["Sauron"], 1, c, .5);


    c.add(name_map["Gandalf"], 0);
    bn.set_variable_probability(name_map["Sauron"], 0, c, .9);
    bn.set_variable_probability(name_map["Sauron"], 1, c, .1);

    c.add(name_map["Gandalf"], 1);
    bn.set_variable_probability(name_map["Sauron"], 0, c, .9);
    bn.set_variable_probability(name_map["Sauron"], 1, c, .1);

    // Bayesian network can be iterated over with a for loop
    for (auto &var: bn) {
        std::cout << var.name() << '\n';
        std::cout << var.table() << '\n';
    }
}
