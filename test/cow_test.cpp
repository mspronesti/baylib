//
// Created by paolo on 02/09/21.
//

#include <gtest/gtest.h>
#include <baylib/probability/condition.hpp>
#include <baylib/smile_utils/smile_utils.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>
#include <baylib/inference/adaptive_importance_sampling.hpp>

#define THREADS std::thread::hardware_concurrency()
#define SAMPLES 10000
#define MEMORY 500*(std::pow(2,30))
#define TOLERANCE 0.05

#define CHECK_NO_COPY  ASSERT_EQ(std::addressof(net5["Income"].table()[c][0]), std::addressof(net5["Assets"].table()[c][0]))
#define CHECK_COPY  ASSERT_NE(std::addressof(net5["Income"].table()[c][0]), std::addressof(net5["Assets"].table()[c][0]))

using namespace bn::inference;

class cow_tests : public ::testing::Test {
protected:
    bn::bayesian_network<double> bn;

    cow_tests()= default;

    ~cow_tests() override= default;

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

TEST_F(cow_tests, cow_flat){
    bn::condition c;
    bn.set_variable_probability("b", 0, c, 0.2);
    bn.set_variable_probability("b", 1, c, 0.8);
    bn.set_variable_probability("c", 0, c, 0.2);
    bn.set_variable_probability("c", 1, c, 0.8);

    const auto & a1 = bn["c"].table();
    const auto & a2 = bn["b"].table();
    const auto & a3 = a1[c];
    const auto & a4 = a2.at(c);
    a1.flat();

    ASSERT_EQ(std::addressof(a3), std::addressof(a4));
}

TEST_F(cow_tests, cow_parser){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = bn::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    //https://repo.bayesfusion.com/network/permalink?net=Large+BNs%2FLink.xdsl
    auto net6 = bn::xdsl_parser<double>().deserialize("../../examples/xdsl/Link.xdsl");
    bn::condition c1;

    const auto& e1 = net5["Income"].table();
    const auto& e2 = net5["Assets"].table();

    ASSERT_EQ(std::addressof(e1[c1][0]), std::addressof(e2[c1][0]));

    bn::condition c2;
    bn::condition c3;
    c2.add("N58_d_f", 0);
    c2.add("N58_d_m", 0);
    c3.add("N57_d_f", 0);
    c3.add("N57_d_m", 0);
    const auto& e3 = net6["N58_d_g"].table();
    const auto& e4 = net6["N57_d_g"].table();
    ASSERT_EQ(std::addressof(e3[c2][0]), std::addressof(e4[c3][0]));
}


TEST_F(cow_tests, cow_inference){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = bn::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");

    bn::condition c;
    auto logic = logic_sampling<double>(SAMPLES, MEMORY);
    auto gibbs = gibbs_sampling<double>(SAMPLES, THREADS);
    auto likely = likelihood_weighting<double>(SAMPLES, THREADS);
    auto adaptive = adaptive_importance_sampling<double>(SAMPLES, MEMORY);
    std::vector<inference_algorithm<double>*> algorithms = {&gibbs, &logic, &likely, &adaptive};
    const auto& e1 = net5["Income"].table();
    const auto& e2 = net5["Assets"].table();

    for (auto algorithm : algorithms){
        CHECK_NO_COPY;
        algorithm->make_inference(net5);
        CHECK_NO_COPY;
    }
}

TEST_F(cow_tests, cow_bn_actions){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = bn::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    bn::condition c;
    CHECK_NO_COPY;

    net5.add_variable("HELLO WORLD", {"HELLO", "WORLD"}); CHECK_NO_COPY;
    net5.add_dependency("HELLO WORLD", "Income"); CHECK_NO_COPY;
    net5.remove_dependency("HELLO WORLD", "Income"); CHECK_NO_COPY;
    net5.index_of("Income"); CHECK_NO_COPY;
    net5.children_of("Income"); CHECK_NO_COPY;
    net5.is_root("Income"); CHECK_NO_COPY;
    net5.has_dependency("Income", "Assets"); CHECK_NO_COPY;
    net5.parents_of("Income"); CHECK_NO_COPY;
    for (auto var: bn){}; CHECK_NO_COPY;
    sampling_order(net5); CHECK_NO_COPY;
    bn::markov_blanket(net5, net5["Income"]); CHECK_NO_COPY;
    net5.set_variable_probability("HELLO WORLD", 0, c, 0.2); CHECK_NO_COPY;
    net5.set_variable_probability("Income", 0, c, 0); CHECK_COPY;
}

TEST_F(cow_tests, cow_variable_actions){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = bn::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    bn::condition c;
    CHECK_NO_COPY;

    net5["Income"].states(); CHECK_NO_COPY;
    net5["Income"].id(); CHECK_NO_COPY;
    net5["Income"].has_state("Income"); CHECK_NO_COPY;
    net5["Income"].has_state("Hello World"); CHECK_NO_COPY;
    net5["Income"].name(); CHECK_NO_COPY;
    cpt_filled_out(net5["Income"]); CHECK_NO_COPY;
    net5["Income"].parents_info.names(); CHECK_NO_COPY;
    net5["Income"].parents_info.add("HELLO WORLD", 10); CHECK_NO_COPY;
    net5["Income"].set_probability(0, c, 0.99); CHECK_COPY;
}

TEST_F(cow_tests, cow_copy_network_not_cpt){
    auto net5 = bn::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    auto net6 = net5;
    bn::condition c;
    ASSERT_EQ(std::addressof(net6["Income"].table()[c][0]), std::addressof(net5["Income"].table()[c][0]));
    ASSERT_EQ(std::addressof(net6["Income"].table()[c][0]), std::addressof(net5["Assets"].table()[c][0]));
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
