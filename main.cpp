//
// Created by paolo on 04/07/2021.
//

#include "graphs/DAG.h"
#include "inference/logic_sampling.hpp"
#include "networks/BayesianNet.h"
#include "tools/COW.h"
#include "tools/thread_pool.hpp"
#include "tools/VariableNodeMap.h"
#include "probability/CPT.h"
#include "parser/BNReader.h"
#include "inference/logic_sampling.hpp"

void test_simulation_source(){
    auto ls = bn::logic_sampling<float>();
    std::vector<float> probabilities = {0.2, 0.8};
    std::vector<bn::bcvec> parents = {};
    bn::bcvec result(1000);
    auto res = ls.simulate_node_agnostic(probabilities, parents, result);
    std::cout << res.first << " " << res.second << "\n";
}

void test_simulation_child(){

}

int main(){
    test_simulation_source();
}