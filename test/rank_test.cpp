//
// Created by elle on 01/08/21.
//

#include <gtest/gtest.h>
#define private public // trick to test private members
#include "../inference/logic_sampling.hpp"

TEST(test_rank, _test){
   auto bn = std::make_shared<bn::bayesian_network<double>>();
   bn->add_variable("a");
   bn->add_variable("b");
   bn->add_variable("c");
   bn->add_variable("d");
   bn->add_variable("e");

   bn->add_dependency("a", "c");
   bn->add_dependency("a", "e");
   bn->add_dependency("b", "c");
   bn->add_dependency("b", "e");
   bn->add_dependency("c", "d");
   bn->add_dependency("d", "e");

   ASSERT_TRUE(bn->is_root("a"));
   ASSERT_TRUE(bn->is_root("b"));

   bn::logic_sampling ls(bn);
   auto rank = ls.graph_rank();

   int expected[] = {0, 0, 1, 2, 3};
   std::uint8_t  i = 0;
   for(auto & r : rank) ASSERT_EQ(r.second, expected[i++]);
}

TEST(test_rank, test_from_book){
    auto bn = std::make_shared<bn::bayesian_network<double>>();
    bn->add_variable("1");
    bn->add_variable("2");
    bn->add_variable("3");
    bn->add_variable("4");
    bn->add_variable("5");
    bn->add_variable("6");
    bn->add_variable("7");

    bn->add_dependency("1", "2");
    bn->add_dependency("1", "3");
    bn->add_dependency("2", "4");
    bn->add_dependency("2", "5");
    bn->add_dependency("2", "6");
    bn->add_dependency("3", "2");
    bn->add_dependency("3", "5");
    bn->add_dependency("3", "6");
    bn->add_dependency("4", "7");
    bn->add_dependency("5", "7");
    bn->add_dependency("6", "4");
    bn->add_dependency("6", "5");


    ASSERT_TRUE(bn->is_root("1"));
    ASSERT_FALSE(bn->is_root("7"));

    bn::logic_sampling ls(bn);
    auto rank = ls.graph_rank();

    int expected[] = {0, 2, 1, 4, 4, 3, 5};
    std::uint8_t  i = 0;
    for(auto & r : rank) ASSERT_EQ(r.second, expected[i++]);
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}