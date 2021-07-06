//
// Created by paolo on 06/07/2021.
//

#include "../graph/BayesianNet.h"

void test_near(double real, double expected,double eps=0.01){
    assert(std::abs(expected-real) < eps);
}

void testFile(){
    BayesianNet net("../xml_files/Coma.xdsl");
    assert(net.name_map["MetastCancer"] == 0);
    assert(net.bnode_vec[0].getName() == "MetastCancer");
    assert(net.bnode_vec[0].getParents().empty());
    assert(net.bnode_vec[1].getParents().size() == 1);
    assert(net.bnode_vec[3].getName() == "Coma");
    assert(net.bnode_vec[3].getProbabilities().size1() == 2);
    assert(net.bnode_vec[3].getProbabilities().size2() == 4);
    cout << net.bnode_vec[3].getProbabilities()(0, 0);
    test_near(net.bnode_vec[3].getProbabilities()(0, 0), .8);
    test_near(net.bnode_vec[3].getProbabilities()(0, 1), .8);
    test_near(net.bnode_vec[3].getProbabilities()(1, 1), .2);
}
