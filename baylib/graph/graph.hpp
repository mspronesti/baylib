//
// Created by elle on 01/08/21.
//

#ifndef BAYESIAN_INFERRER_GRAPH_HPP
#define BAYESIAN_INFERRER_GRAPH_HPP

#include <boost/graph/adjacency_list.hpp>

namespace bn {
    template <typename Probability>
    struct variable {
        unsigned int id{};
        std::string name;
        std::vector<std::string> states; // names of states
        std::map<std::string, Probability> marginal_probs;

        bool operator < (const variable & other) const { return id < other.id; }
    };

    template <typename Probability>
    using graph = boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS, variable<Probability>>;

    template <typename Probability>
    using vertex = typename graph<Probability>::vertex_descriptor;


} // namespace bn

#endif //BAYESIAN_INFERRER_GRAPH_HPP
