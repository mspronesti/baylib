//
// Created by paolo on 04/07/2021.
//

#include "bnode.h"

#include <utility>

bnode::bnode(const string &name, const list<string> &events, const list<string> &parents,
             const ub::matrix<double> &probabilities){
    this->name = name;
    this->events = events;
    this->parents = parents;
    this->probabilities = probabilities;
}

const string &bnode::getName() const {
    return name;
}

const list<string> &bnode::getEvents() const {
    return events;
}

const list<string> &bnode::getParents() const {
    return parents;
}

const ub::matrix<double> &bnode::getProbabilities() const {
    return probabilities;
}
