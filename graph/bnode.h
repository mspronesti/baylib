//
// Created by paolo on 04/07/2021.
//

#ifndef GPUTEST_BNODE_H
#define GPUTEST_BNODE_H


#include <string>
#include <list>
#include <memory>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

using namespace std;
namespace ub = boost::numeric::ublas;
class bnode {
private:
    string name;
    list<string> events{};
    list<string> parents{};
    ub::matrix<double> probabilities;
public:
    bnode(const string &name, const list<string> &events, const list<string> &parents,
          const ub::matrix<double> &probabilities);

    const string &getName() const;

    const list<string> &getEvents() const;

    const list<string> &getParents() const;

    const ub::matrix<double> &getProbabilities() const;

};


#endif //GPUTEST_BNODE_H
