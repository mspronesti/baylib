//
// Created by elle on 01/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>

class bnet_tests : public ::testing::Test {
protected:
    enum {
        A,
        B,
        C,
        D,
        E
    };

    bn::bayesian_network<bn::random_variable<double>> bn;

    bnet_tests()= default;

    ~bnet_tests() override= default;

    void SetUp() override {
        //   a
        //  /  \
        // v    v
        // b    c <--- e
        // \   /
        //  \ /
        //   v
        //   d
        bn.add_variable();
        bn.add_variable();
        bn.add_variable();
        bn.add_variable();
        bn.add_variable();

        bn.add_dependency(A, B);
        bn.add_dependency(A, C);
        bn.add_dependency(C, D);
        bn.add_dependency(B, D);
        bn.add_dependency(E, C);
    }
};


TEST_F(bnet_tests, test_names){
    std::vector<int> e{A, B, C, D, E};
    std::uint8_t i = 0;
    for(auto & var : bn)
        EXPECT_EQ(e[i++], var.id());
}

TEST_F(bnet_tests, test_root){
    ASSERT_TRUE(bn.is_root(A));
    ASSERT_TRUE(bn.is_root(E));
    ASSERT_FALSE(bn.is_root(B));
    ASSERT_FALSE(bn.is_root(C));
    ASSERT_FALSE(bn.is_root(D));
}

TEST_F(bnet_tests, test_dependency){
    ASSERT_TRUE(bn.has_dependency(A, B));
    ASSERT_TRUE(bn.has_dependency(A, C));
    ASSERT_TRUE(bn.has_dependency(B, D));
    ASSERT_TRUE(bn.has_dependency(C, D));

    ASSERT_FALSE(bn.has_dependency(B, A));
    ASSERT_FALSE(bn.has_dependency(C, A));
    ASSERT_FALSE(bn.has_dependency(D, B));
    ASSERT_FALSE(bn.has_dependency(D, C));
}

TEST_F(bnet_tests, test_not_dag){
    ASSERT_THROW(bn.add_dependency(D, A), std::logic_error);
    ASSERT_THROW(bn.add_dependency(D, E), std::logic_error);
    ASSERT_NO_THROW(bn.add_dependency(E, D));
}

TEST_F(bnet_tests, test_children){
    auto children = bn.children_of(A);
    auto a_id = bn[B].id();
    auto b_id = bn[C].id();

    ASSERT_EQ(children[0], a_id);
    ASSERT_EQ(children[1], b_id);
}

TEST_F(bnet_tests, test_parents){
    auto parents = bn.parents_of(D);

    auto b_id = bn[B].id();
    auto c_id = bn[C].id();

    ASSERT_EQ(parents[0], c_id);
    ASSERT_EQ(parents[1], b_id);
    ASSERT_TRUE(bn.has_dependency(B, D));
    ASSERT_TRUE(bn.has_dependency(C, D));
}

TEST_F(bnet_tests, test_invalid_varname){
    ASSERT_ANY_THROW(bn[E + 3]);
}

TEST_F(bnet_tests, test_invalid_edge){
    ASSERT_ANY_THROW(bn.add_dependency(A, E + 3));
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
