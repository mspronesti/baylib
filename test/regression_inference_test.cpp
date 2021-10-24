//
// Created by paolo on 07/09/21.
//

#include <gtest/gtest.h>
#include <baylib/smile_utils/smile_utils.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>
#include <baylib/inference/rejection_sampling.hpp>
#include <baylib/inference/adaptive_importance_sampling.hpp>

#define THREADS std::thread::hardware_concurrency()
#define SAMPLES 10000
#define MEMORY 500*(std::pow(2,30))
#define TOLERANCE 0.05

using namespace baylib::inference;
using Probability = double;

template<typename Probability, class Variable>
        std::vector<baylib::marginal_distribution<Probability>> get_results(const baylib::bayesian_net<Variable> &bn){
            std::vector<baylib::marginal_distribution<Probability>> results{
                logic_sampling<baylib::bayesian_net<Variable>>(bn, SAMPLES, MEMORY).make_inference(),
                gibbs_sampling<baylib::bayesian_net<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                likelihood_weighting<baylib::bayesian_net<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                rejection_sampling<baylib::bayesian_net<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                adaptive_importance_sampling<baylib::bayesian_net<Variable>>(bn, SAMPLES, MEMORY).make_inference()
            };
            return results;
        }

        template<typename Probability, class Variable>
        std::vector<baylib::marginal_distribution<Probability>> get_results_deterministic(const baylib::bayesian_net<Variable> &bn){
            std::vector<baylib::marginal_distribution<Probability>> results{
                logic_sampling<baylib::bayesian_net<Variable>>(bn, SAMPLES, MEMORY).make_inference(),
                likelihood_weighting<baylib::bayesian_net<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                rejection_sampling<baylib::bayesian_net<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                adaptive_importance_sampling<baylib::bayesian_net<Variable>>(bn, SAMPLES, MEMORY).make_inference()
            };
            return results;
        }



void test(const std::string& file_name, bool deterministic){

    auto net = baylib::xdsl_parser<double>().deserialize(file_name);
    auto marginals = !deterministic ? get_results<Probability>(net) : get_results_deterministic<Probability>(net);

    for(uint i = 0; i < net.number_of_variables(); i++){
        for(uint j = 1; j < marginals.size(); j++){
            for(uint k = 0; k < marginals[j][i].size(); k++){
                ASSERT_NEAR(marginals[0][i][k], marginals[j][i][k], TOLERANCE);
            }
        }
    }
}

TEST(regression_test, coma){
    test("../../examples/xdsl/Coma.xdsl", false);
}

TEST(regression_test, asia){
    test("../../examples/xdsl/AsiaDiagnosis.xdsl", true);
}

TEST(regression_test, animals){
    test("../../examples/xdsl/Animals.xdsl", true);
}

TEST(regression_test, venture){
    test("../../examples/xdsl/VentureBNExpanded.xdsl", false);
}

TEST(regression_test, credit){
    test("../../examples/xdsl/Credit.xdsl", false);
}

TEST(regression_test, hail){
    test("../../examples/xdsl/Hailfinder2.5.xdsl", true);
}

TEST(regression_test, link){
    test("../../examples/xdsl/Link.xdsl", true);
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}