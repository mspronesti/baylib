//
// Created by paolo on 06/07/2021.
//

#include "../graph/BayesianNet.h"
#include <boost/graph/adjacency_list.hpp>

void test_near(double real, double expected, double eps=0.001){
    assert(std::abs(expected-real) < eps);
}

void testFile(){
    BayesianNet net("../xml_files/Coma.xdsl");
    assert(net.name_map["MetastCancer"] == 0);
    assert(net.bnode_vec[0].name == "MetastCancer");
    assert(net.bnode_vec[0].parents.empty());
    assert(net.bnode_vec[1].parents.size() == 1);
    assert(net.bnode_vec[3].name == "Coma");
    assert(net.bnode_vec[3].probabilities.size1() == 2);
    assert(net.bnode_vec[3].probabilities.size2() == 4);
    test_near(net.bnode_vec[3].probabilities(0, 0), .8);
    test_near(net.bnode_vec[3].probabilities(0, 1), .8);
    test_near(net.bnode_vec[3].probabilities(1, 1), .2);
    assert(boost::edge(0, 1, net.network).second);
    assert(boost::edge(0, 2, net.network).second);
    assert(!boost::edge(1, 0, net.network).second);
    assert(!boost::edge(2, 0, net.network).second);
    assert(boost::edge(1, 3, net.network).second);
    assert(!boost::edge(1, 2, net.network).second);
    assert(boost::edge(2, 3, net.network).second);
    assert(boost::edge(1, 4, net.network).second);
}
