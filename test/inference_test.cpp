//
// Created by elle on 05/08/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_network.hpp>
#include <baylib/smile_utils/smile_utils.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>
#include <baylib/inference/rejection_sampling.hpp>
#include <baylib/inference/adaptive_importance_sampling.hpp>


#define THREADS std::thread::hardware_concurrency()
#define SAMPLES 20000
#define MEMORY 500*(std::pow(2,30))
#define TOLERANCE 0.05

using namespace bn::inference;
using Probability = double;

class inference_tests : public ::testing::Test{
protected:

    inference_tests() = default;
    ~inference_tests() override = default;

    /*
    void SetUp() override{
        auto logic = std::make_shared<logic_sampling<Probability>>(SAMPLES, MEMORY);
        auto gibbs = std::make_shared<gibbs_sampling<Probability>>(SAMPLES, THREADS);
        auto likely = std::make_shared<likelihood_weighting<Probability>>(SAMPLES, THREADS);
        auto rejection = std::make_shared<rejection_sampling<Probability, std::default_random_engine>>(SAMPLES, THREADS);
        auto adaptive =  std::make_shared<adaptive_importance_sampling<Probability>>(SAMPLES, MEMORY);

        algorithms = { gibbs,
                       logic,
                       likely,
                       rejection,
                       adaptive
                       };

        algorithms_det = { logic,
                           likely,
                           rejection,
                           adaptive
                          };
    }*/
};

template<typename Probability, class Variable>
std::vector<bn::marginal_distribution<Probability>> get_results(const bn::bayesian_network<Variable> &bn){
    std::vector<bn::marginal_distribution<Probability>> results{
        logic_sampling<Probability>(SAMPLES, MEMORY).make_inference(bn),
        gibbs_sampling<Probability>(SAMPLES, THREADS).make_inference(bn),
        likelihood_weighting<Probability>(SAMPLES, THREADS).make_inference(bn),
        rejection_sampling<Probability>(SAMPLES, THREADS).make_inference(bn),
        adaptive_importance_sampling<Probability>(SAMPLES, MEMORY).make_inference(bn)
    };
    return results;
}

template<typename Probability, class Variable>
        std::vector<bn::marginal_distribution<Probability>> get_results_deterministic(const bn::bayesian_network<Variable> &bn){
            std::vector<bn::marginal_distribution<Probability>> results{
                logic_sampling<Probability>(SAMPLES, MEMORY).make_inference(bn),
                likelihood_weighting<Probability>(SAMPLES, THREADS).make_inference(bn),
                rejection_sampling<Probability>(SAMPLES, THREADS).make_inference(bn),
                adaptive_importance_sampling<Probability>(SAMPLES, MEMORY).make_inference(bn)
            };
            return results;
        }

/**
 * Basic test on a quite small network
 */
TEST_F(inference_tests, big_bang_Coma){

    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::xdsl_parser<Probability>().deserialize("../../examples/xdsl/Coma.xdsl");
    //bn::inference::logic_sampling<Probability> alg = bn::inference::logic_sampling<Probability>(SAMPLES, MEMORY);
    auto n_map = bn::make_name_map(net1);
    for (bn::marginal_distribution<double>& result: get_results<double>(net1)){
        ASSERT_NEAR(result[n_map["MetastCancer"]][0], .2, TOLERANCE);
        ASSERT_NEAR(result[n_map["MetastCancer"]][1], .8, TOLERANCE);

        ASSERT_NEAR(result[n_map["IncrSerCal"]][0], .32, TOLERANCE);
        ASSERT_NEAR(result[n_map["IncrSerCal"]][1], .68, TOLERANCE);

        ASSERT_NEAR(result[n_map["Coma"]][0], .32, TOLERANCE);
        ASSERT_NEAR(result[n_map["Coma"]][1], .68, TOLERANCE);

        ASSERT_NEAR(result[n_map["BrainTumor"]][0], .08, TOLERANCE);
        ASSERT_NEAR(result[n_map["BrainTumor"]][1], .92, TOLERANCE);

        ASSERT_NEAR(result[n_map["SevHeadaches"]][0], .62, TOLERANCE);
        ASSERT_NEAR(result[n_map["SevHeadaches"]][1], .38, TOLERANCE);
        }
    }

    /**
     * Test with networks with non-binary variables
     */

    TEST_F(inference_tests, big_bang_VentureBNExpanded){

        //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FVentureBNExpanded.xdsl
        auto net2 = bn::xdsl_parser<Probability>().deserialize("../../examples/xdsl/VentureBNExpanded.xdsl");
        auto n_map = bn::make_name_map(net2);

        for(auto& result : get_results<double>(net2)){

            ASSERT_NEAR(result[n_map["Success"]][0], .2, TOLERANCE);
            ASSERT_NEAR(result[n_map["Success"]][1], .8, TOLERANCE);

            ASSERT_NEAR(result[n_map["Economy"]][0], .2, TOLERANCE);
            ASSERT_NEAR(result[n_map["Economy"]][1], .7, TOLERANCE);
            ASSERT_NEAR(result[n_map["Economy"]][2], .1, TOLERANCE);

            ASSERT_NEAR(result[n_map["Forecast"]][0], .23, TOLERANCE);
            ASSERT_NEAR(result[n_map["Forecast"]][1], .3, TOLERANCE);
            ASSERT_NEAR(result[n_map["Forecast"]][2], .47, TOLERANCE);
        }
    }

    /**
     * Test on medium-size bayesian network
     */
    TEST_F(inference_tests, big_bang_Credit){

        //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
        auto net3 = bn::xdsl_parser<Probability>().deserialize("../../examples/xdsl/Credit.xdsl");
        auto n_map = bn::make_name_map(net3);

        for(auto& result : get_results<double>(net3)){

            ASSERT_NEAR(result[n_map["PaymentHistory"]][0], .25, TOLERANCE);
            ASSERT_NEAR(result[n_map["PaymentHistory"]][1], .25, TOLERANCE);
            ASSERT_NEAR(result[n_map["PaymentHistory"]][2], .25, TOLERANCE);
            ASSERT_NEAR(result[n_map["PaymentHistory"]][3], .25, TOLERANCE);

            ASSERT_NEAR(result[n_map["WorkHistory"]][0], .25, TOLERANCE);
            ASSERT_NEAR(result[n_map["WorkHistory"]][1], .25, TOLERANCE);
            ASSERT_NEAR(result[n_map["WorkHistory"]][2], .25, TOLERANCE);
            ASSERT_NEAR(result[n_map["WorkHistory"]][3], .25, TOLERANCE);

            ASSERT_NEAR(result[n_map["Reliability"]][0], .45, TOLERANCE);
            ASSERT_NEAR(result[n_map["Reliability"]][1], .55, TOLERANCE);

            ASSERT_NEAR(result[n_map["Debit"]][0], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Debit"]][1], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Debit"]][2], .33, TOLERANCE);

            ASSERT_NEAR(result[n_map["Income"]][0], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Income"]][1], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Income"]][2], .33, TOLERANCE);

            ASSERT_NEAR(result[n_map["RatioDebInc"]][0], .47, TOLERANCE);
            ASSERT_NEAR(result[n_map["RatioDebInc"]][1], .53, TOLERANCE);

            ASSERT_NEAR(result[n_map["Assets"]][0], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Assets"]][1], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Assets"]][2], .33, TOLERANCE);

            ASSERT_NEAR(result[n_map["Worth"]][0], .59, TOLERANCE);
            ASSERT_NEAR(result[n_map["Worth"]][1], .22, TOLERANCE);
            ASSERT_NEAR(result[n_map["Worth"]][2], .19, TOLERANCE);

            ASSERT_NEAR(result[n_map["Profession"]][0], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Profession"]][1], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Profession"]][2], .33, TOLERANCE);

            ASSERT_NEAR(result[n_map["FutureIncome"]][0], .68, TOLERANCE);
            ASSERT_NEAR(result[n_map["FutureIncome"]][1], .32, TOLERANCE);

            ASSERT_NEAR(result[n_map["Age"]][0], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Age"]][1], .33, TOLERANCE);
            ASSERT_NEAR(result[n_map["Age"]][2], .33, TOLERANCE);

            ASSERT_NEAR(result[n_map["CreditWorthiness"]][0], .54, TOLERANCE);
            ASSERT_NEAR(result[n_map["CreditWorthiness"]][1], .46, TOLERANCE);
        }
    }

    /**
     * Test on mixture between absolute and non absolute probabilities
     */

    TEST_F(inference_tests, big_bang_Asia){

        //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FAsiaDiagnosis.xdsl
        auto net4 = bn::xdsl_parser<Probability>().deserialize("../../examples/xdsl/AsiaDiagnosis.xdsl");
        auto n_map = bn::make_name_map(net4);

        for(auto& result : get_results_deterministic<double>(net4)){

            ASSERT_NEAR(result[n_map["Tuberculosis"]][0], .99, TOLERANCE);
            ASSERT_NEAR(result[n_map["Tuberculosis"]][1], .01, TOLERANCE);

            ASSERT_NEAR(result[n_map["TbOrCa"]][0], .94, TOLERANCE);
            ASSERT_NEAR(result[n_map["TbOrCa"]][1], .06, TOLERANCE);

            ASSERT_NEAR(result[n_map["XRay"]][0], .89, TOLERANCE);
            ASSERT_NEAR(result[n_map["XRay"]][1], .11, TOLERANCE);

            ASSERT_NEAR(result[n_map["Dyspnea"]][0], .56, TOLERANCE);
            ASSERT_NEAR(result[n_map["Dyspnea"]][1], .44, TOLERANCE);

            ASSERT_NEAR(result[n_map["Bronchitis"]][0], .55, TOLERANCE);
            ASSERT_NEAR(result[n_map["Bronchitis"]][1], .45, TOLERANCE);
        }
    }

    /**
     * Test a quite large network
     */

    TEST_F(inference_tests, big_bang_Hail){

        https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FHailfinder2.5.xdsl
        auto net5 = bn::xdsl_parser<Probability>().deserialize("../../examples/xdsl/Hailfinder2.5.xdsl");
        auto n_map = bn::make_name_map(net5);

        for(auto& result : get_results_deterministic<double>(net5)){

            ASSERT_NEAR(result[n_map["R5Fcst"]][0], 0.25, TOLERANCE);
            ASSERT_NEAR(result[n_map["R5Fcst"]][1], 0.44, TOLERANCE);
            ASSERT_NEAR(result[n_map["R5Fcst"]][2], 0.31, TOLERANCE);
            ASSERT_NEAR(result[n_map["CompPlFcst"]][0], 0.41 ,TOLERANCE);
            ASSERT_NEAR(result[n_map["CompPlFcst"]][1], 0.36 ,TOLERANCE);
            ASSERT_NEAR(result[n_map["CompPlFcst"]][2], 0.24 ,TOLERANCE);
        }
    }

    /**
     * Test on a large network (~ 200 000)
     */

    TEST_F(inference_tests, big_bang_Link){

        //https://repo.bayesfusion.com/network/permalink?net=Large+BNs%2FLink.xdsl
        auto net6 = bn::xdsl_parser<Probability>().deserialize("../../examples/xdsl/Link.xdsl");
        auto n_map = bn::make_name_map(net6);

        for(auto& result : get_results_deterministic<double>(net6)){

            ASSERT_NEAR(result[n_map["N59_d_g"]][0], 0., TOLERANCE);
            ASSERT_NEAR(result[n_map["N59_d_g"]][1], 0.01, TOLERANCE);
            ASSERT_NEAR(result[n_map["N59_d_g"]][2], 0.99, TOLERANCE);

            ASSERT_NEAR(result[n_map["D0_56_a_f"]][0], 0.25, TOLERANCE);
            ASSERT_NEAR(result[n_map["D0_56_a_f"]][1], 0.25, TOLERANCE);
            ASSERT_NEAR(result[n_map["D0_56_a_f"]][2], 0.25, TOLERANCE);
            ASSERT_NEAR(result[n_map["D0_56_a_f"]][3], 0.25, TOLERANCE);

            ASSERT_NEAR(result[n_map["D0_56_d_p"]][0], 0., TOLERANCE);
            ASSERT_NEAR(result[n_map["D0_56_d_p"]][1], 1, TOLERANCE);
        }
    }


    int main(int argc, char** argv){
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }