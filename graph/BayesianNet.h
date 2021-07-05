//
// Created by paolo on 04/07/2021.
//

#ifndef GPUTEST_BAYESIANNET_H
#define GPUTEST_BAYESIANNET_H


#include <vector>
#include <boost/graph/adjacency_list.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> Graph;

class BayesianNet {
private:
    Graph graph;
public:
    explicit BayesianNet(const char *file_name);
};


#endif //GPUTEST_BAYESIANNET_H
