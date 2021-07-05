//
// Created by paolo on 04/07/2021.
//

#include "bnode.h"

#include <utility>

bnode::bnode(string name, const list<string> &events, const list<string> &parents,
             const ub::matrix<int> &probabilities) : name(std::move(name)), events(events), parents(parents),
                                                     probabilities(probabilities) {}
