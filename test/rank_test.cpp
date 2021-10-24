//
// Created by elle on 01/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_utils.hpp>
#include <baylib/smile_utils/smile_utils.hpp>


TEST(test_rank, _test){
   auto bn = baylib::bayesian_net<baylib::named_random_variable<double>>();
   ulong a = bn.add_variable("a", std::vector<std::string>{"s1", "s2"});
   ulong b = bn.add_variable("b", std::vector<std::string>{"s1", "s2"});
   ulong c = bn.add_variable("c", std::vector<std::string>{"s1", "s2"});
   ulong d = bn.add_variable("d", std::vector<std::string>{"s1", "s2"});
   ulong e = bn.add_variable("e", std::vector<std::string>{"s1", "s2"});

   bn.add_dependency(a, c);
   bn.add_dependency(a, e);
   bn.add_dependency(b, c);
   bn.add_dependency(b, e);
   bn.add_dependency(c, d);
   bn.add_dependency(d, e);

   ASSERT_TRUE(bn.is_root(a));
   ASSERT_TRUE(bn.is_root(b));

    auto order = baylib::sampling_order(bn);
    int exp_order[] = {0, 1, 2, 3, 4};
    for (int j = 0; j < order.size(); ++j)
        ASSERT_EQ(order[j], exp_order[j]);
}

TEST(test_rank, test_from_book){
    auto bn = baylib::bayesian_net<baylib::named_random_variable<double>>();
    auto v1 = bn.add_variable("1", std::vector<std::string>{"1", "2", "3"});
    auto v2 = bn.add_variable("2", std::vector<std::string>{"1", "2", "3"});
    auto v3 = bn.add_variable("3", std::vector<std::string>{"1", "2", "3"});
    auto v4 = bn.add_variable("4", std::vector<std::string>{"1", "2", "3"});
    auto v5 = bn.add_variable("5", std::vector<std::string>{"1", "2", "3"});
    auto v6 = bn.add_variable("6", std::vector<std::string>{"1", "2", "3"});
    auto v7 = bn.add_variable("7", std::vector<std::string>{"1", "2", "3"});

    bn.add_dependency(v1, v2);
    bn.add_dependency(v1, v3);
    bn.add_dependency(v2, v4);
    bn.add_dependency(v2, v5);
    bn.add_dependency(v2, v6);
    bn.add_dependency(v3, v2);
    bn.add_dependency(v3, v5);
    bn.add_dependency(v3, v6);
    bn.add_dependency(v4, v7);
    bn.add_dependency(v5, v7);
    bn.add_dependency(v6, v4);
    bn.add_dependency(v6, v5);

    ASSERT_TRUE(bn.is_root(v1));
    ASSERT_FALSE(bn.is_root(v7));

    auto order = baylib::sampling_order(bn);
    int exp_order[] = {0, 2, 1, 5, 3, 4, 6};
    for (int j = 0; j < order.size(); ++j)
        ASSERT_EQ(order[j], exp_order[j]);
}



int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
