//
// Created by elle on 05/08/21.
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

class inference_tests : public ::testing::Test{
protected:

    std::vector<std::shared_ptr<inference_algorithm<float>>> algorithms;

    inference_tests() = default;
    ~inference_tests() override = default;

    void SetUp() override{
        std::shared_ptr<inference_algorithm<float>> logic = std::make_shared<logic_sampling<float>>(MEMORY, SAMPLES);
        std::shared_ptr<inference_algorithm<float>> gibbs = std::make_shared<gibbs_sampling<float>>(SAMPLES, THREADS);
        std::shared_ptr<inference_algorithm<float>> likely = std::make_shared<likelihood_weighting<float>>(SAMPLES, THREADS);
        algorithms = {gibbs, logic, likely};
    }
};

// Basic starting test
TEST_F(inference_tests, big_bang_Coma){

    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::xdsl_parser<float>().deserialize("../../examples/xdsl/Coma.xdsl");

    for(const auto& sampling : algorithms){
        auto result = sampling->make_inference(net1);
        ASSERT_NEAR(result[net1.index_of("MetastCancer")][0], .2, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("MetastCancer")][1], .8, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("IncrSerCal")][0], .32, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("IncrSerCal")][1], .68, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("Coma")][0], .32, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("Coma")][1], .68, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("BrainTumor")][0], .08, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("BrainTumor")][1], .92, TOLERANCE);

        ASSERT_NEAR(result[net1.index_of("SevHeadaches")][0], .62, TOLERANCE);
        ASSERT_NEAR(result[net1.index_of("SevHeadaches")][1], .38, TOLERANCE);
    }
}

// Test on non binary variables
TEST_F(inference_tests, big_bang_VentureBNExpanded){

    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FVentureBNExpanded.xdsl
    auto net2 = bn::xdsl_parser<float>().deserialize("../../examples/xdsl/VentureBNExpanded.xdsl");


    for(const auto& sampling : algorithms){
        auto result = sampling->make_inference(net2);

        ASSERT_NEAR(result[net2.index_of("Success")][0], .2, TOLERANCE);
        ASSERT_NEAR(result[net2.index_of("Success")][1], .8, TOLERANCE);

        ASSERT_NEAR(result[net2.index_of("Economy")][0], .2, TOLERANCE);
        ASSERT_NEAR(result[net2.index_of("Economy")][1], .7, TOLERANCE);
        ASSERT_NEAR(result[net2.index_of("Economy")][2], .1, TOLERANCE);

        ASSERT_NEAR(result[net2.index_of("Forecast")][0], .23, TOLERANCE);
        ASSERT_NEAR(result[net2.index_of("Forecast")][1], .3, TOLERANCE);
        ASSERT_NEAR(result[net2.index_of("Forecast")][2], .47, TOLERANCE);\
        }
    }

    // Test on medium size bayesian network
    TEST_F(inference_tests, big_bang_Credit){

        //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
        auto net3 = bn::xdsl_parser<float>().deserialize("../../examples/xdsl/Credit.xdsl");

        for(const auto& sampling : algorithms){
            auto result = sampling->make_inference(net3);

            ASSERT_NEAR(result[net3.index_of("PaymentHistory")][0], .25, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("PaymentHistory")][1], .25, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("PaymentHistory")][2], .25, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("PaymentHistory")][3], .25, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("WorkHistory")][0], .25, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("WorkHistory")][1], .25, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("WorkHistory")][2], .25, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("WorkHistory")][3], .25, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("Reliability")][0], .45, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Reliability")][1], .55, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("Debit")][0], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Debit")][1], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Debit")][2], .33, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("Income")][0], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Income")][1], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Income")][2], .33, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("RatioDebInc")][0], .47, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("RatioDebInc")][1], .53, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("Assets")][0], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Assets")][1], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Assets")][2], .33, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("Worth")][0], .59, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Worth")][1], .22, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Worth")][2], .19, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("Profession")][0], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Profession")][1], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Profession")][2], .33, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("FutureIncome")][0], .68, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("FutureIncome")][1], .32, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("Age")][0], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Age")][1], .33, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("Age")][2], .33, TOLERANCE);

            ASSERT_NEAR(result[net3.index_of("CreditWorthiness")][0], .54, TOLERANCE);
            ASSERT_NEAR(result[net3.index_of("CreditWorthiness")][1], .46, TOLERANCE);
        }
    }

    // Test on mixture between absolute and non absolute probabilities
    TEST_F(inference_tests, big_bang_Asia){

        //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FAsiaDiagnosis.xdsl
        auto net4 = bn::xdsl_parser<float>().deserialize("../../examples/xdsl/AsiaDiagnosis.xdsl");

        for(const auto& sampling : algorithms){
            auto result = sampling->make_inference(net4);

            ASSERT_NEAR(result[net4.index_of("Tuberculosis")][0], .99, TOLERANCE);
            ASSERT_NEAR(result[net4.index_of("Tuberculosis")][1], .01, TOLERANCE);

            ASSERT_NEAR(result[net4.index_of("TbOrCa")][0], .94, TOLERANCE);
            ASSERT_NEAR(result[net4.index_of("TbOrCa")][1], .06, TOLERANCE);

            ASSERT_NEAR(result[net4.index_of("XRay")][0], .89, TOLERANCE);
            ASSERT_NEAR(result[net4.index_of("XRay")][1], .11, TOLERANCE);

            ASSERT_NEAR(result[net4.index_of("Dyspnea")][0], .56, TOLERANCE);
            ASSERT_NEAR(result[net4.index_of("Dyspnea")][1], .44, TOLERANCE);

            ASSERT_NEAR(result[net4.index_of("Bronchitis")][0], .55, TOLERANCE);
            ASSERT_NEAR(result[net4.index_of("Bronchitis")][1], .45, TOLERANCE);
        }
    }

    // Test on Large network
    TEST_F(inference_tests, big_bang_Hail){

        https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FHailfinder2.5.xdsl
        auto net5 = bn::xdsl_parser<float>().deserialize("../../examples/xdsl/Hailfinder2.5.xdsl");

        for(const auto& sampling : algorithms){
            auto result = sampling->make_inference(net5);

            ASSERT_NEAR(result[net5.index_of("R5Fcst")][0], 0.25, TOLERANCE);
            ASSERT_NEAR(result[net5.index_of("R5Fcst")][1], 0.44, TOLERANCE);
            ASSERT_NEAR(result[net5.index_of("R5Fcst")][2], 0.31, TOLERANCE);
            ASSERT_NEAR(result[net5.index_of("CompPlFcst")][0], 0.41 ,TOLERANCE);
            ASSERT_NEAR(result[net5.index_of("CompPlFcst")][1], 0.36 ,TOLERANCE);
            ASSERT_NEAR(result[net5.index_of("CompPlFcst")][2], 0.24 ,TOLERANCE);
        }
    }

    // Test on very large network
    TEST_F(inference_tests, big_bang_Link){

        //https://repo.bayesfusion.com/network/permalink?net=Large+BNs%2FLink.xdsl
        auto net6 = bn::xdsl_parser<float>().deserialize("../../examples/xdsl/Link.xdsl");

        for(const auto& sampling : algorithms){
            auto result = sampling->make_inference(net6);

            ASSERT_NEAR(result[net6.index_of("N59_d_g")][0], 0., TOLERANCE);
            ASSERT_NEAR(result[net6.index_of("N59_d_g")][1], 0.01, TOLERANCE);
            ASSERT_NEAR(result[net6.index_of("N59_d_g")][2], 0.99, TOLERANCE);

            ASSERT_NEAR(result[net6.index_of("D0_56_a_f")][0], 0.25, TOLERANCE);
            ASSERT_NEAR(result[net6.index_of("D0_56_a_f")][1], 0.25, TOLERANCE);
            ASSERT_NEAR(result[net6.index_of("D0_56_a_f")][2], 0.25, TOLERANCE);
            ASSERT_NEAR(result[net6.index_of("D0_56_a_f")][3], 0.25, TOLERANCE);

            ASSERT_NEAR(result[net6.index_of("D0_56_d_p")][0], 0., TOLERANCE);
            ASSERT_NEAR(result[net6.index_of("D0_56_d_p")][1], 1, TOLERANCE);
        }
    }


    int main(int argc, char** argv){
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }