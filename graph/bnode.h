#ifndef GPUTEST_BNODE_H
#define GPUTEST_BNODE_H

#include <string>
#include <list>
#include <memory>
#include <utility>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

namespace ub = boost::numeric::ublas;

struct bnode {
    std::string name;
    std::list<std::string> events;
    std::list<std::string> parents;
    ub::matrix<double> probabilities;

    bnode(std::string name,  std::list<std::string> events,
            std::list<std::string> parents, const ub::matrix<double> &probabilities):
          name(std::move(name)), events(std::move(events)), parents(std::move(parents)), probabilities(probabilities){}
};


#endif //GPUTEST_BNODE_H
