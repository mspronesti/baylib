//
// Created by elle on 01/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/algorithm.hpp>

using rank_t = std::map<bn::vertex<double>, int>;

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

   rank_t rank_map = bn::graph_rank(bn);

   int expected[] = {0, 0, 1, 2, 3};
   std::uint8_t  i = 0;
   for(auto & r : rank_map) ASSERT_EQ(r.second, expected[i++]);

    auto order = bn::rank_order(bn);
    int exp_order[] = {0, 1, 2, 3, 4};
    for (int j = 0; j < order.size(); ++j)
        ASSERT_EQ(order[j], exp_order[j]);
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

    rank_t rank = bn::graph_rank(bn);

    int expected[] = {0, 2, 1, 4, 4, 3, 5};
    std::uint8_t  i = 0;
    for(auto & r : rank) ASSERT_EQ(r.second, expected[i++]);

    auto order = bn::rank_order(bn);
    int exp_order[] = {0, 2, 1, 5, 3, 4, 6};
    for (int j = 0; j < order.size(); ++j)
        ASSERT_EQ(order[j], exp_order[j]);
}



int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}