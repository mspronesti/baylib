//
// Created by elle on 05/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>
#include <baylib/network/bayesian_utils.hpp>
#include <baylib/parser/net_parser.hpp>
#include <baylib/inference/gibbs_sampling.hpp>

#define THREADS 10
#define SAMPLES 10000
#define TOLERANCE 0.05


TEST(gibbs_sampling_tests, big_bang_Coma){
    bn::bayesian_network<double> net1;
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    net1 = bn::net_parser<double>().load_from_xdsl("../../test/xdsl/Coma.xdsl");
    bn::inference::gibbs_sampling<double> sampling(net1);
    sampling.inferenciate(SAMPLES, THREADS);
    auto result = sampling.inference_result();
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

TEST(gibbs_sampling_tests, big_bang_VentureBNExpanded){
    bn::bayesian_network<float> net2;
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FVentureBNExpanded.xdsl
    net2 = bn::net_parser<float>().load_from_xdsl("../../test/xdsl/VentureBNExpanded.xdsl");
    bn::inference::gibbs_sampling<float> sampling(net2);
    sampling.inferenciate(SAMPLES, THREADS);
    auto result = sampling.inference_result();

    ASSERT_NEAR(result[net2.index_of("Success")][0], .2, TOLERANCE);
    ASSERT_NEAR(result[net2.index_of("Success")][1], .8, TOLERANCE);

    ASSERT_NEAR(result[net2.index_of("Economy")][0], .2, TOLERANCE);
    ASSERT_NEAR(result[net2.index_of("Economy")][1], .7, TOLERANCE);
    ASSERT_NEAR(result[net2.index_of("Economy")][2], .1, TOLERANCE);

    ASSERT_NEAR(result[net2.index_of("Forecast")][0], .23, TOLERANCE);
    ASSERT_NEAR(result[net2.index_of("Forecast")][1], .3, TOLERANCE);
    ASSERT_NEAR(result[net2.index_of("Forecast")][2], .47, TOLERANCE);
}

TEST(gibbs_sampling_tests, big_bang_Credit){
    bn::bayesian_network<float> net3;
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    net3 = bn::net_parser<float>().load_from_xdsl("../../test/xdsl/Credit.xdsl");
    bn::inference::gibbs_sampling<float> sampling(net3);
    sampling.inferenciate(SAMPLES, THREADS);
    auto result = sampling.inference_result();

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

TEST(gibbs_sampling_tests, big_bang_Asia){
    bn::bayesian_network<float> net4;
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FAsiaDiagnosis.xdsl
    net4 = bn::net_parser<float>().load_from_xdsl("../../test/xdsl/AsiaDiagnosis.xdsl");

    bn::inference::gibbs_sampling<float> sampling(net4);
    sampling.inferenciate(SAMPLES, THREADS);
    auto result = sampling.inference_result();

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

TEST(gibbs_sampling_tests, big_bang_Hail){
    https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FHailfinder2.5.xdsl
    auto net5 = bn::net_parser<float>().load_from_xdsl("../../test/xdsl/Hailfinder2.5.xdsl");

    bn::inference::gibbs_sampling<float> sampling(net5);
    sampling.inferenciate(SAMPLES, THREADS);
    auto result = sampling.inference_result();

    ASSERT_NEAR(result[net5.index_of("R5Fcst")][0], 0.25, TOLERANCE);
    ASSERT_NEAR(result[net5.index_of("R5Fcst")][1], 0.44, TOLERANCE);
    ASSERT_NEAR(result[net5.index_of("R5Fcst")][2], 0.31, TOLERANCE);
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}