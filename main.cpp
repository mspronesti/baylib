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

int main(){
    std::shared_ptr<BayesianNetwork<double>> BN = std::make_shared<BayesianNetwork<double>>();
    BNReader<double>().loadNetworkFromFile("../xml_files/Coma.xdsl", BN);
}