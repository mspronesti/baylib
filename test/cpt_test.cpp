//
// Created by elle on 02/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/condition.hpp>
#include <baylib/network/bayesian_utils.hpp>

class cpt_tests : public ::testing::Test {
protected:
    bn::bayesian_network<double> bn;

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
        bn.add_variable("a", {"T", "F"});
        bn.add_variable("b", {"T", "F"});
        bn.add_variable("c", {"T", "F"});
        bn.add_variable("d", {"T", "F"});

        bn.add_dependency("b", "a");
        bn.add_dependency("c", "a");
        bn.add_dependency("a", "d");
    }
};

enum {
    F = 0,
    T = 1
};

TEST_F(cpt_tests, test_root_1) {
    ASSERT_FALSE(bn::cpt_filled_out(bn["b"]));

    bn::condition c; // empty condition
    bn.set_variable_probability("b", T, c, .02);
    bn.set_variable_probability("b", F, c, 1 - .02);

    auto& b_cpt = bn["b"].table();
    ASSERT_EQ(b_cpt[c][F], 1 - .02);
    ASSERT_EQ(b_cpt[c][T],  .02);
    ASSERT_TRUE(bn::cpt_filled_out(bn["b"]));
}


TEST_F(cpt_tests, test_root_2) {
    bn::condition c; // empty condition
    bn.set_variable_probability("c", T, c, .002);
    bn.set_variable_probability("c", F, c, 1 - .002);

    auto &c_cpt = bn["c"].table();
    ASSERT_EQ(c_cpt[c][F], 1 - .002);
    ASSERT_EQ(c_cpt[c][T], .002);
}

TEST_F(cpt_tests, test_parents) {
    bn::condition c;
    c.add("b", 1);
    c.add("c", 1);

    // can't set a probability > 1
    ASSERT_ANY_THROW( bn.set_variable_probability("a", T, c, 1.5));

    bn.set_variable_probability("a", T, c, .9); // P(a = 1 | b = 1, c = 1) = 0.9
    bn.set_variable_probability("a", F, c, 1 - .9); // P(a = 0 | b = 1, c = 1) = 0.1

    auto& a_cpt = bn["a"].table();
    ASSERT_EQ(a_cpt[c][F], 1 - .9);
    ASSERT_EQ(a_cpt[c][T],  .9);

    bn::condition c1;
    c1.add("b", 0);
    c1.add("c", 1);
    bn.set_variable_probability("a", T, c1, .5);
    bn.set_variable_probability("a", F, c1, 1-.5);

    ASSERT_EQ(a_cpt[c1][F], 1 - .5);
    ASSERT_EQ(a_cpt[c1][T],  .5);

    // (0,0) and (1,0) missing
    ASSERT_FALSE(bn::cpt_filled_out(bn["a"]));

    c1.clear();
    c1.add("b", 0);
    c1.add("c", 0);
    bn.set_variable_probability("a", T, c1, .25);
    bn.set_variable_probability("a", F, c1, 1-.25);

    ASSERT_EQ(a_cpt[c1][F], 1 - .25);
    ASSERT_EQ(a_cpt[c1][T],  .25);

    // (1,0) missing
    ASSERT_FALSE(bn::cpt_filled_out(bn["a"]));
    c1.clear();
    c1.add("b", 1);
    c1.add("c", 0);
    bn.set_variable_probability("a", T, c1, .023);
    bn.set_variable_probability("a", F, c1, 1-.023);

    ASSERT_TRUE(bn::cpt_filled_out(bn["a"]));

    c1.clear();
    c1.add("a", 1);
    bn.set_variable_probability("d", T, c1, 0.5);
    bn.set_variable_probability("d", F, c1, 1 - 0.5);

    auto& b_cpt = bn["d"].table();
    ASSERT_EQ(b_cpt[c1][F], 1 - .5);
    ASSERT_EQ(b_cpt[c1][T],  .5);

    std::cout << a_cpt;
}


TEST_F(cpt_tests, test_no_parent) {
    bn::condition c;
    c.add("a", 1);
    c.add("c", 1);

    // "b" is root, hence doesn't have any parent!
    ASSERT_ANY_THROW(bn.set_variable_probability("b", T, c, 0.5));
    // "c" ain't "d"'s parent
    ASSERT_ANY_THROW(bn.set_variable_probability("d", T, c, 0.5));
}

TEST_F(cpt_tests, test_wrong_line_sum) {
    bn::condition c;
    c.add("a", 1);

    bn.set_variable_probability("d", T, c, 0.123);
    // missing probability for state F and case a = 0
    ASSERT_FALSE(bn::cpt_filled_out(bn["d"]));
    bn.set_variable_probability("d", F, c, 1 - .123);

    c.add("a", 0);
    bn.set_variable_probability("d", T, c, .3);
    bn.set_variable_probability("d", F, c, .65);

    // row sum is != 1
    ASSERT_FALSE(bn::cpt_filled_out(bn["d"]));
    // now it sums to 1
    bn.set_variable_probability("d", F, c, 1 - .3);
    ASSERT_TRUE(bn::cpt_filled_out(bn["d"]));
}

TEST_F(cpt_tests, test_cpt_equal){
    bn::condition c;
    bn.set_variable_probability("b", 0, c, 0.2);
    bn.set_variable_probability("b", 0, c, 0.8);
    bn.set_variable_probability("c", 0, c, 0.2);
    bn.set_variable_probability("c", 0, c, 0.8);

    ASSERT_TRUE(bn["c"].table() == bn["b"].table());

    bn.set_variable_probability("b", 0, c, 0.4);

    ASSERT_FALSE(bn["c"].table() == bn["b"].table());
}
TEST_F(cpt_tests, cow){
    bn::condition c;
    bn.set_variable_probability("b", 0, c, 0.2);
    bn.set_variable_probability("b", 1, c, 0.8);
    bn.set_variable_probability("c", 0, c, 0.2);
    bn.set_variable_probability("c", 1, c, 0.8);

    const auto & a1 = bn["c"].table();
    const auto & a2 = bn["b"].table();
    const auto & a3 = a1[c];
    const auto & a4 = a2.at(c);

    ASSERT_EQ(std::addressof(a3), std::addressof(a4));

    bn.set_variable_probability("c", 0, c, 0.3);
    bn.set_variable_probability("c", 1, c, 0.7);

    const auto & a5 = bn["c"].table();
    const auto & a6 = bn["b"].table();
    const auto & a7 = a1[c];
    const auto & a8 = a2.at(c);

    ASSERT_NE(std::addressof(a7), std::addressof(a8));
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
