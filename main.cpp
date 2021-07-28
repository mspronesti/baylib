//
// Created by paolo on 04/07/2021.
//

#include "graphs/DAG.h"
#include "inference/logic_sampling.hpp"
#include "networks/BayesianNet.h"
#include "parser/BNReader.h"

void test_simulation_source(){
    auto ls = bn::logic_sampling<float>();
    std::vector<float> probabilities = {0.2, 0.8};
    std::vector<std::shared_ptr<bn::bcvec>> parents = {};
    auto vec = ls.simulate_node(probabilities, parents);
    auto res = ls.compute_result_binary(*vec);
    std::cout << res.first << " " << res.second << "\n";
}


void test_simulation_chain(){
    int n = 10000;
    auto ls = bn::logic_sampling<float>();
    std::pair<int, int> res_pair;
    std::vector<float> prob_source = {0.2, 0.8};
    std::vector<std::shared_ptr<bn::bcvec>> parent_res = {};
    std::shared_ptr<bn::bcvec> res = ls.simulate_node(prob_source, parent_res, n);
    res_pair = ls.compute_result_binary(*res);
    std::cout << "MetastCancer:" << res_pair.first << " " << res_pair.second << "\n";
    parent_res = {res};
    prob_source = {0.8, 0.2, 0.2, 0.8};
    std::shared_ptr<bn::bcvec> res2 = ls.simulate_node(prob_source, parent_res, n);
    res_pair = ls.compute_result_binary(*res2);
    std::cout << "IncreasedSerumCalcium:" << res_pair.first << " " << res_pair.second << "\n";
    prob_source = {0.2, 0.08, 0.05, 0.95};
    std::shared_ptr<bn::bcvec> res3 = ls.simulate_node(prob_source, parent_res, n);
    res_pair = ls.compute_result_binary(*res3);
    std::cout << "BrainTumor:" << res_pair.first << " " << res_pair.second << "\n";
    parent_res = {res2, res3};
    prob_source = {0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.05, 0.095};
    std::shared_ptr<bn::bcvec> res4 = ls.simulate_node(prob_source, parent_res, n);
    res_pair = ls.compute_result_binary(*res4);
    std::cout << "COMA:" << res_pair.first << " " << res_pair.second << "\n";

}
int main(){
    test_simulation_chain();
}