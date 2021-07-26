#include "graph/DAG.h"
#include "inference/logic_sampling.hpp"
#include "network/bayesian_network.h"
#include "tools/COW.h"
#include "tools/thread_pool.hpp"
#include "tools/VariableNodeMap.h"
#include "probability/CPT.h"
#include "parser/BNReader.h"
#include "inference/logic_sampling.hpp"
#include "network/bayesian_network.h"

int main(){
    std::shared_ptr<bn::bayesian_network<double>> BN = std::make_shared<bn::bayesian_network<double>>();
    BNReader<double>().loadNetworkFromFile("../examples/Coma.xdsl", BN);
}