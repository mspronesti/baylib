//
// Created by paolo on 02/09/21.
//

#include <gtest/gtest.h>
#include <baylib/smile_utils/smile_utils.hpp>
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

#define CHECK_NO_COPY  ASSERT_EQ(std::addressof(net5[n_map["Income"]].table()[c][0]), std::addressof(net5[n_map["Assets"]].table()[c][0]))
#define CHECK_COPY  ASSERT_NE(std::addressof(net5[n_map["Income"]].table()[c][0]), std::addressof(net5[n_map["Assets"]].table()[c][0]))

using namespace baylib::inference;

class cow_tests : public ::testing::Test {
protected:
    typedef baylib::bayesian_net<baylib::named_random_variable<double>> named_bayesian_network;
    enum {
        A,
        B,
        C,
        D,
        E
    };


    baylib::bayesian_net<baylib::random_variable<double>> bn;

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
        bn.add_variable();
        bn.add_variable();
        bn.add_variable();
        bn.add_variable();

        bn.add_dependency(B, A);
        bn.add_dependency(C, A);
        bn.add_dependency(A, D);
    }
};


TEST_F(cow_tests, cow_parser){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    //https://repo.bayesfusion.com/network/permalink?net=Large+BNs%2FLink.xdsl
    auto net6 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Link.xdsl");
    baylib::condition c1;

    auto n_map = baylib::make_name_map(net5);

    const auto& e1 = net5[n_map["Income"]].table();
    const auto& e2 = net5[n_map["Assets"]].table();

    ASSERT_EQ(std::addressof(e1[c1][0]), std::addressof(e2[c1][0]));

    n_map = baylib::make_name_map(net6);
    baylib::condition c2;
    baylib::condition c3;
    c2.add(n_map["N58_d_f"], 0);
    c2.add(n_map["N58_d_m"], 0);
    c3.add(n_map["N57_d_f"], 0);
    c3.add(n_map["N57_d_m"], 0);
    const auto& e3 = net6[n_map["N58_d_g"]].table();
    const auto& e4 = net6[n_map["N57_d_g"]].table();
    ASSERT_EQ(std::addressof(e3[c2][0]), std::addressof(e4[c3][0]));
}


TEST_F(cow_tests, cow_inference){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    auto n_map = baylib::make_name_map(net5);

    baylib::condition c;
    auto logic = logic_sampling<named_bayesian_network>(net5, SAMPLES, MEMORY);
    auto gibbs = gibbs_sampling<named_bayesian_network>(net5, SAMPLES, THREADS);
    auto likely = likelihood_weighting<named_bayesian_network>(net5, SAMPLES, THREADS);
    auto adaptive = adaptive_importance_sampling<named_bayesian_network>(net5, SAMPLES, MEMORY);
    std::vector<inference_algorithm<named_bayesian_network>*> algorithms = {&gibbs, &logic, &likely, &adaptive};

    auto name_map = baylib::make_name_map(net5);
    const auto& e1 = net5[name_map["Income"]].table();
    const auto& e2 = net5[name_map["Assets"]].table();

    for (auto algorithm : algorithms){
        CHECK_NO_COPY;
        algorithm->make_inference();
        CHECK_NO_COPY;
    }
}

TEST_F(cow_tests, cow_bn_actions){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    auto n_map = baylib::make_name_map(net5);
    baylib::condition c;
    CHECK_NO_COPY;

    auto hello = net5.add_variable("HELLO WORLD", std::vector<std::string>{"HELLO", "WORLD"}); CHECK_NO_COPY;
    net5.add_dependency(hello, n_map["Income"]); CHECK_NO_COPY;
    net5.remove_dependency(hello, n_map["Income"]); CHECK_NO_COPY;
    net5.children_of(n_map["Income"]); CHECK_NO_COPY;
    net5.is_root(n_map["Income"]); CHECK_NO_COPY;
    net5.has_dependency(n_map["Income"], n_map["Assets"]); CHECK_NO_COPY;
    net5.parents_of(n_map["Income"]); CHECK_NO_COPY;
    for (auto var: bn){}; CHECK_NO_COPY;
    sampling_order(net5); CHECK_NO_COPY;
    baylib::markov_blanket(net5, n_map["Income"]); CHECK_NO_COPY;
    net5.set_variable_probability(hello, 0, c, 0.2); CHECK_NO_COPY;
    net5.set_variable_probability(n_map["Income"], 0, c, 0); CHECK_COPY;
}

TEST_F(cow_tests, cow_variable_actions){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    auto n_map = baylib::make_name_map(net5);
    baylib::condition c;
    CHECK_NO_COPY;

    net5[n_map["Income"]].states(); CHECK_NO_COPY;
    net5[n_map["Income"]].id(); CHECK_NO_COPY;
    net5[n_map["Income"]].has_state("Income"); CHECK_NO_COPY;
    net5[n_map["Income"]].has_state("Hello World"); CHECK_NO_COPY;
    net5[n_map["Income"]].name(); CHECK_NO_COPY;
    cpt_filled_out(net5, n_map["Income"]); CHECK_NO_COPY;
    net5[n_map["Income"]].set_probability(0, c, 0.99); CHECK_COPY;
}

TEST_F(cow_tests, cow_copy_network_not_cpt){
    auto net5 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    auto net6 = net5;
    auto n_map = baylib::make_name_map(net5);
    baylib::condition c;
    ASSERT_EQ(std::addressof(net6[n_map["Income"]].table()[c][0]), std::addressof(net5[n_map["Assets"]].table()[c][0]));
    ASSERT_EQ(std::addressof(net6[n_map["Income"]].table()[c][0]), std::addressof(net5[n_map["Assets"]].table()[c][0]));
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
