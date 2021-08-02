//
// Created by elle on 01/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>

class bnet_tests : public ::testing::Test {
protected:
    bn::bayesian_network<double> bn;

    bnet_tests()= default;

    ~bnet_tests() override= default;

    void SetUp() override {
        //   a
        //  .  .
        // .    .
        // b    c . . e
        // .    .
        //  .  .
        //    d
        bn.add_variable("a");
        bn.add_variable("b");
        bn.add_variable("c");
        bn.add_variable("d");
        bn.add_variable("e");

        bn.add_dependency("a", "b");
        bn.add_dependency("a", "c");
        bn.add_dependency("c", "d");
        bn.add_dependency("b", "d");
        bn.add_dependency("e", "c");
    }
};


TEST_F(bnet_tests, test_names){
    std::vector<std::string> e{"a", "b", "c", "d", "e"};
    std::uint8_t i = 0;
    for(auto & a : bn.variables())
        EXPECT_EQ(e[i++], a.name);
}

TEST_F(bnet_tests, test_root){
    ASSERT_TRUE(bn.is_root("a"));
    ASSERT_TRUE(bn.is_root("e"));
    ASSERT_FALSE(bn.is_root("b"));
    ASSERT_FALSE(bn.is_root("c"));
    ASSERT_FALSE(bn.is_root("d"));
}

TEST_F(bnet_tests, test_dependency){
    ASSERT_TRUE(bn.conditional_dependency("a", "b"));
    ASSERT_TRUE(bn.conditional_dependency("a", "c"));
    ASSERT_TRUE(bn.conditional_dependency("b", "d"));
    ASSERT_TRUE(bn.conditional_dependency("c", "d"));

    ASSERT_FALSE(bn.conditional_dependency("b", "a"));
    ASSERT_FALSE(bn.conditional_dependency("c", "a"));
    ASSERT_FALSE(bn.conditional_dependency("d", "b"));
    ASSERT_FALSE(bn.conditional_dependency("d", "c"));
}

TEST_F(bnet_tests, test_not_dag){
    // would be not longer DAG!
    ASSERT_ANY_THROW(bn.add_dependency("d", "a"));
}

TEST_F(bnet_tests, test_children){
    auto children = bn.children_of("a");
    auto a_id = bn.getVariable("b").id;
    auto b_id = bn.getVariable("c").id;

    ASSERT_EQ(children[0], a_id);
    ASSERT_EQ(children[1], b_id);
}

TEST_F(bnet_tests, test_parents){
    auto parents = bn.parents_of("d");

    auto b_id = bn.getVariable("b").id;
    auto c_id = bn.getVariable("c").id;

    ASSERT_EQ(parents[0], c_id);
    ASSERT_EQ(parents[1], b_id);
}

TEST_F(bnet_tests, test_invalid_varname){
    ASSERT_ANY_THROW(bn.getVariable("pippo"));
}

TEST_F(bnet_tests, test_invalid_edge){
    ASSERT_ANY_THROW(bn.add_dependency("a", "pippo"));
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}