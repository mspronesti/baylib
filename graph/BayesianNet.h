//
// Created by paolo on 04/07/2021.
//

#ifndef GPUTEST_BAYESIANNET_H
#define GPUTEST_BAYESIANNET_H


#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include "bnode.h"

using Graph =  boost::adjacency_list<boost::vecS, boost::vecS>;


class BayesianNet {
public:
    Graph network; // adjacency list of the network
    std::map<std::string, int> name_map; // map: node_name -> graph_index/vector_index
    std::vector<bnode> bnode_vec; // vector of bnodes structs
    explicit BayesianNet(const std::string &file_name);
};


#endif //GPUTEST_BAYESIANNET_H
