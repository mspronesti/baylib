//
// Created by paolo on 07/09/21.
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

std::vector<std::unique_ptr<bn::inference::inference_algorithm<double>>> get_alg(){
    auto alg = std::vector<std::unique_ptr<inference_algorithm<double>>>();
    alg.emplace_back(std::move(std::make_unique<logic_sampling<double>>(MEMORY, SAMPLES)));
    alg.emplace_back(std::move(std::make_unique<likelihood_weighting<double>>(SAMPLES, THREADS)));
    alg.emplace_back(std::move(std::make_unique<gibbs_sampling<double>>(SAMPLES, THREADS)));
    return alg;
}

std::vector<std::unique_ptr<bn::inference::inference_algorithm<double>>> get_alg_not_det(){
    auto alg = std::vector<std::unique_ptr<inference_algorithm<double>>>();
    alg.emplace_back(std::move(std::make_unique<logic_sampling<double>>(MEMORY, SAMPLES)));
    alg.emplace_back(std::move(std::make_unique<likelihood_weighting<double>>(SAMPLES, THREADS)));
    return alg;
}


void test(const std::string& file_name, bool deterministic){
    auto algorithms = !deterministic? get_alg() : get_alg_not_det();
    auto net = bn::xdsl_parser<double>().deserialize(file_name);
    std::vector<bn::marginal_distribution<double>> marginals(0, bn::marginal_distribution<double>(net.begin(), net.end()));
    for(auto& algorithm: algorithms)
        marginals.emplace_back(algorithm->make_inference(net));

    for(int i=0; i<net.number_of_variables(); i++){
        for(int j=1; i<marginals.size(); i++){
            for(int k=0; k<marginals[j][i].size(); k++){
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