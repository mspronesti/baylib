//
// Created by elle on 02/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>
#include <baylib/network/probability/condition.hpp>

class cpt_tests : public ::testing::Test {
protected:
    bn::bayesian_network<double> bn;

    cpt_tests()= default;

    ~cpt_tests() override= default;

    void SetUp() override {
        //
        // b    c
        // .    .
        //  .  .
        //    a
        //    .
        //    .
        //    d

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
  bn::condition c; // empty condition
  bn.set_variable_probability("b", T, c, .02);
  bn.set_variable_probability("b", F, c, 1 - .02);

  ASSERT_EQ(bn["b"].table()[c][F], 1 - .02);
  ASSERT_EQ(bn["b"].table()[c][T],  .02);
}


TEST_F(cpt_tests, test_root_2) {
    bn::condition c; // empty condition
    bn.set_variable_probability("c", T, c, .002);
    bn.set_variable_probability("c", F, c, 1 - .002);

    ASSERT_EQ(bn["c"].table()[c][F], 1 - .002);
    ASSERT_EQ(bn["c"].table()[c][T],  .002);
}

TEST_F(cpt_tests, test_parents) {
    bn::condition c;
    c.add("b", 1);
    c.add("c", 1);

    bn.set_variable_probability("a", T, c, .9); // P(a = 1 | b = 1, c = 1) = 0.9
    bn.set_variable_probability("a", F, c, 1 - .9); // P(a = 0 | b = 1, c = 1) = 0.1
    ASSERT_EQ(bn["a"].table()[c][F], 1 - .9);
    ASSERT_EQ(bn["a"].table()[c][T],  .9);

    bn::condition c1;
    c1.add("b", 0);
    c1.add("c", 1);
    bn.set_variable_probability("a", T, c1, .5);
    bn.set_variable_probability("a", F, c1, 1-.5);
    ASSERT_EQ(bn["a"].table()[c1][F], 1 - .5);
    ASSERT_EQ(bn["a"].table()[c1][T],  .5);


    c1.clear();
    c1.add("a", 1);
    bn.set_variable_probability("d", T, c1, 0.5);
    bn.set_variable_probability("d", F, c1, 1 - 0.5);

    ASSERT_EQ(bn["d"].table()[c1][F], 1 - .5);
    ASSERT_EQ(bn["d"].table()[c1][T],  .5);

}



int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
