//
// Created by elle on 02/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/condition.hpp>
#include <baylib/network/bayesian_utils.hpp>
#include <baylib/probability/icpt.hpp>

class cpt_tests : public ::testing::Test {
protected:
    bn::bayesian_network<bn::random_variable<double>> bn;
    ulong var_A;
    ulong var_B;
    ulong var_C;
    ulong var_D;


    cpt_tests()= default;

    ~cpt_tests() override= default;

    void SetUp() override {
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

        // 2 states variables
        var_A = bn.add_variable((ulong)2);
        var_B = bn.add_variable((ulong)2);
        var_C= bn.add_variable((ulong)2);
        var_D = bn.add_variable((ulong)2);

        bn.add_dependency(var_B, var_A);
        bn.add_dependency(var_C, var_A);
        bn.add_dependency(var_A, var_D);
    }
};

enum {
    F = 0,
    T = 1
};

TEST_F(cpt_tests, test_root_1) {
    ASSERT_FALSE(bn::cpt_filled_out(bn, var_B));

    bn::condition c; // empty condition
    bn.set_variable_probability(var_B, T, c, .02);
    bn.set_variable_probability(var_B, F, c, 1 - .02);

    auto& b_cpt = bn[var_B].table();
    ASSERT_EQ(b_cpt[c][F], 1 - .02);
    ASSERT_EQ(b_cpt[c][T],  .02);
    ASSERT_TRUE(bn::cpt_filled_out(bn, var_B));
}


TEST_F(cpt_tests, test_root_2) {
    bn::condition c; // empty condition
    bn.set_variable_probability(var_C, T, c, .002);
    bn.set_variable_probability(var_C, F, c, 1 - .002);

    auto &c_cpt = bn[var_C].table();
    ASSERT_EQ(c_cpt[c][F], 1 - .002);
    ASSERT_EQ(c_cpt[c][T], .002);
}

TEST_F(cpt_tests, test_parents) {
    bn::condition c;
    c.add(var_B, 1);
    c.add(var_C, 1);

    // can't set a probability > 1
    ASSERT_ANY_THROW( bn.set_variable_probability(var_A, T, c, 1.5));

    bn.set_variable_probability(var_A, T, c, .9); // P(a = 1 | b = 1, c = 1) = 0.9
    bn.set_variable_probability(var_A, F, c, 1 - .9); // P(a = 0 | b = 1, c = 1) = 0.1

    auto& a_cpt = bn[var_A].table();
    ASSERT_EQ(a_cpt[c][F], 1 - .9);
    ASSERT_EQ(a_cpt[c][T],  .9);

    bn::condition c1;
    c1.add(var_B, 0);
    c1.add(var_C, 1);
    bn.set_variable_probability(var_A, T, c1, .5);
    bn.set_variable_probability(var_A, F, c1, 1-.5);

    ASSERT_EQ(a_cpt[c1][F], 1 - .5);
    ASSERT_EQ(a_cpt[c1][T],  .5);

    // (0,0) and (1,0) missing
    ASSERT_FALSE(bn::cpt_filled_out(bn, var_A));

    c1.clear();
    c1.add(var_B, 0);
    c1.add(var_C, 0);
    bn.set_variable_probability(var_A, T, c1, .25);
    bn.set_variable_probability(var_A, F, c1, 1-.25);

    ASSERT_EQ(a_cpt[c1][F], 1 - .25);
    ASSERT_EQ(a_cpt[c1][T],  .25);

    // (1,0) missing
    ASSERT_FALSE(bn::cpt_filled_out(bn, var_A));
    c1.clear();
    c1.add(var_B, 1);
    c1.add(var_C, 0);
    bn.set_variable_probability(var_A, T, c1, .023);
    bn.set_variable_probability(var_A, F, c1, 1-.023);

    ASSERT_TRUE(bn::cpt_filled_out(bn, var_A));

    c1.clear();
    c1.add(var_A, 1);
    bn.set_variable_probability(var_D, T, c1, 0.5);
    bn.set_variable_probability(var_D, F, c1, 1 - 0.5);

    auto& b_cpt = bn[var_D].table();
    ASSERT_EQ(b_cpt[c1][F], 1 - .5);
    ASSERT_EQ(b_cpt[c1][T],  .5);

}


TEST_F(cpt_tests, test_no_parent) {
    bn::condition c;
    c.add(var_A, 1);
    c.add(var_C, 1);

    // "b" is root, hence doesn't have any parent!
    ASSERT_ANY_THROW(bn.set_variable_probability(var_B, T, c, 0.5));
    // "c" ain't "d"'s parent
    ASSERT_ANY_THROW(bn.set_variable_probability(var_D, T, c, 0.5));
}

TEST_F(cpt_tests, test_wrong_line_sum) {
    bn::condition c;
    c.add(var_A, 1);

    bn.set_variable_probability(var_D, T, c, 0.123);
    // missing probability for state F and case a = 0
    ASSERT_FALSE(bn::cpt_filled_out(bn, var_D));
    bn.set_variable_probability(var_D, F, c, 1 - .123);

    c.add(var_A, 0);
    bn.set_variable_probability(var_D, T, c, .3);
    bn.set_variable_probability(var_D, F, c, .65);

    // row sum is != 1
    ASSERT_FALSE(bn::cpt_filled_out(bn, var_D));
    // now it sums to 1
    bn.set_variable_probability(var_D, F, c, 1 - .3);
    ASSERT_TRUE(bn::cpt_filled_out(bn, var_D));
}

TEST_F(cpt_tests, test_cpt_equal){
    bn::condition c;
    bn.set_variable_probability(var_B, 0, c, 0.2);
    bn.set_variable_probability(var_B, 0, c, 0.8);
    bn.set_variable_probability(var_C, 0, c, 0.2);
    bn.set_variable_probability(var_C, 0, c, 0.8);

    ASSERT_TRUE(bn[var_C].table() == bn[var_B].table());

    bn.set_variable_probability(var_B, 0, c, 0.4);

    ASSERT_FALSE(bn[var_C].table() == bn[var_B].table());
}
TEST_F(cpt_tests, cow){
    bn::condition c;
    bn.set_variable_probability(var_B, 0, c, 0.2);
    bn.set_variable_probability(var_B, 1, c, 0.8);
    bn.set_variable_probability(var_C, 0, c, 0.2);
    bn.set_variable_probability(var_C, 1, c, 0.8);

    const auto & a1 = bn[var_C].table();
    const auto & a2 = bn[var_B].table();
    const auto & a3 = a1[c];
    const auto & a4 = a2.at(c);

    ASSERT_EQ(std::addressof(a3), std::addressof(a4));

    bn.set_variable_probability(var_C, 0, c, 0.3);
    bn.set_variable_probability(var_C, 1, c, 0.7);

    const auto & a5 = bn[var_C].table();
    const auto & a6 = bn[var_B].table();
    const auto & a7 = a1[c];
    const auto & a8 = a2.at(c);

    ASSERT_NE(std::addressof(a7), std::addressof(a8));
}



int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
