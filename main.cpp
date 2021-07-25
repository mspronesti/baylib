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
    std::vector<float> probabilities = {0.2, 0.8};
    std::vector<bn::bcvec> parents = {};
    bn::bcvec result(1000);
    bn::simulate_node_agnostic(probabilities, parents, result);
}

void test_simulation_child(){

}

int main(){
    test_simulation();
}