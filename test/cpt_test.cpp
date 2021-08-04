//
// Created by elle on 02/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/probability/cpt.hpp>

TEST(test_cpt, test_root_1){
    bn::random_variable<double> var;
    var.states.emplace_back("s1");
    var.states.emplace_back("s2");

    bn::cpt cpt{var};
    for(auto & p :  cpt[{}]) // empty condition
            ASSERT_EQ(0.5, p);
}

TEST(test_cpt, test_root_2){
    bn::random_variable<double> var;
    var.states.emplace_back("s1");
    var.states.emplace_back("s2");
    var.states.emplace_back("s3");

    bn::cpt cpt{var};
    for(auto & p :  cpt[{}]) // empty condition
        ASSERT_EQ(1.0/3, p);
}


TEST(test_cpt, test_entire_graph){

}


