//
// Created by paolo on 03/09/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>
#include <baylib/parser/xdsl_parser.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>


#define THREADS std::thread::hardware_concurrency()
#define SAMPLES 10000
#define MEMORY 500*(std::pow(2,30))
#define TOLERANCE 0.05

using namespace bn::inference;
using Probability = double;


class evidence_test : public ::testing::Test{
protected:
    typedef std::shared_ptr<inference_algorithm<Probability>> algorithm_ptr;
    std::vector<algorithm_ptr> algorithms;

    evidence_test() = default;
    ~evidence_test() override = default;

    void SetUp() override{
        auto logic = std::make_shared<logic_sampling<Probability>>(MEMORY, SAMPLES);
        auto gibbs = std::make_shared<gibbs_sampling<Probability>>(SAMPLES, THREADS);
        auto likely = std::make_shared<likelihood_weighting<Probability>>(SAMPLES, THREADS);

        algorithms = { likely
                     //, gibbs
                     //, logic
        };
    }
};

TEST_F(evidence_test, evidence_coma){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::xdsl_parser<Probability>().deserialize("../../examples/xdsl/Coma.xdsl");

    net1["Coma"].set_as_evidence(0);

    for(const auto& sampling : algorithms){
        auto result = sampling->make_inference(net1);
        ASSERT_NEAR(result[net1.index_of("MetastCancer")][0], .43, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("MetastCancer")][1], .58, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("IncrSerCal")][0], .8, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("IncrSerCal")][1], .2, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("Coma")][0], 1, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("Coma")][1], 0, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("BrainTumor")][0], .2, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("BrainTumor")][1], .8, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("SevHeadaches")][0], .64, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("SevHeadaches")][1], .36, TOLERANCE);
    }

    net1["Coma"].reset_evidence();
    net1["IncrSerCal"].set_as_evidence(0);
    net1["SevHeadaches"].set_as_evidence(1);

    for(const auto& sampling : algorithms){
        auto result = sampling->make_inference(net1);
        ASSERT_NEAR(result[net1.index_of("MetastCancer")][0], .48, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("MetastCancer")][1], .52, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("IncrSerCal")][0], 1, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("IncrSerCal")][1], 0, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("Coma")][0], .8, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("Coma")][1], .2, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("BrainTumor")][0], .07, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("BrainTumor")][1], .93, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("SevHeadaches")][0], 0, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("SevHeadaches")][1], 1, TOLERANCE);
    }
}