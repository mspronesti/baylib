//
// Created by paolo on 04/07/2021.
//

#ifndef GPUTEST_BNODE_H
#define GPUTEST_BNODE_H


#include <string>
#include <list>
#include <memory>
#include <utility>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

using namespace std;
namespace ub = boost::numeric::ublas;
struct bnode {
    string name;
    list<string> events{};
    list<string> parents{};
    ub::matrix<double> probabilities;

    bnode(string name, const list<string> &events, const list<string> &parents,
          const ub::matrix<double> &probabilities):
          name(std::move(name)), events(events), parents(parents), probabilities(probabilities){}
};


#endif //GPUTEST_BNODE_H
