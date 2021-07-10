//
// Created by paolo on 04/07/2021.
//

#ifndef GPUTEST_BAYESIANNET_H
#define GPUTEST_BAYESIANNET_H


#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include "bnode.h"

typedef boost::adjacency_list<boost::vecS, boost::vecS> Graph;
using namespace std;


class BayesianNet {
public:
    Graph network; // adjacency list of the network
    map<string, int> name_map; // map: node_name -> graph_index/vector_index
    vector<bnode> bnode_vec; //
    explicit BayesianNet(const char *file_name);
};


#endif //GPUTEST_BAYESIANNET_H
