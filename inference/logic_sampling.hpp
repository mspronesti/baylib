//
// Created by elle on 22/07/21.
//

#ifndef GPUTEST_LOGIC_SAMPLING_HPP
#define GPUTEST_LOGIC_SAMPLING_HPP

#include <vector>
#include <future>
#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

#include <boost/graph/adjacency_list.hpp>

#include "../network/bayesian_network.h"

namespace bn {
    namespace compute = boost::compute;
    /* aliases */
    /// graph
    using graph = boost::adjacency_list<boost::setS, boost::setS, boost::bidirectionalS>;
    using node = graph::vertex_descriptor;
    /// gpgpu
    using bcvec = boost::compute::vector<int>;
    /// graph ranking
    using rank_t = std::map<node, int>;

    template <typename T>
    class logic_sampling {
    public:
        logic_sampling() {
            // TODO: to be implemented
        };

        logic_sampling(const compute::device &device, const bn::bayesian_network<T> &bn){
            to_boost_graph(bn);
            // TODO: add missing
        }
    private:
        graph g;
        std::vector<node> nodes;

        // private members
        rank_t graph_rank();
        bool exists_edge(node v1, node v2, const graph &g);
        void to_boost_graph(const bn::bayesian_network<T> &bn);
    };

    template <typename T=float >
    std::pair<int, int> simulate_node(std::vector<T> striped_cpt, std::vector<std::future<bcvec>> parents_result, std::promise<bcvec> result);

    /**
    * Applies ranking function to the DAG g
    * @tparam T
    * @return map containing node-rank as key-value couple
    */
    template<typename T>
    rank_t logic_sampling<T>::graph_rank() {
        std::vector<node> roots;
        rank_t ranks{};

        // fill nodes map with 0s
        for(auto & v : nodes)
            ranks[v] = 0;

        // find roots
        graph::vertex_iterator v, vend;
        for (auto vd : boost::make_iterator_range(boost::vertices(g)))
            if(boost::in_degree(vd, g) == 0)
                roots.push_back(vd);

        if(roots.empty())
            throw std::logic_error("No root nodes found in graph.");

        while(!roots.empty()) {
            node curr_node = roots.back();
            roots.pop_back();

            for (auto vd : boost::make_iterator_range(boost::vertices(g))) {
                if (!exists_edge(curr_node, vd, g)) continue;

                if (ranks[curr_node] + 1 > ranks[vd]) {
                    ranks[vd] = ranks[curr_node] + 1;
                    roots.push_back(vd);
                }
            }
        }

        return ranks;
    }


    /**
    * determines whether nodes v1 and v2 are adjacent
    * @tparam T
    * @param v1 : source node
    * @param v2 : destination node
    * @param g  : BGL adjacency list
    * @return true if v1's adjacent v2, false otherwise
    */
    template<typename T>
    bool logic_sampling<T>::exists_edge(bn::node v1, bn::node v2, const bn::graph &g) {
        return boost::edge(v1, v2, g).second;
    }

    /**
    * Extracts needed info from custom graph to build internal
    * graph using BGL
    * @tparam T
    * @param bn : bayesian network
    */
    template <typename T>
    void logic_sampling<T>::to_boost_graph(const bn::bayesian_network<T> &bn){
        // TODO: to be implemented
    }


} // namespace bn


#endif //GPUTEST_LOGIC_SAMPLING_HPP
