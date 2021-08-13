//
// Created by elle on 02/08/21.
//

#ifndef BAYLIB_BAYESIAN_UTILS_HPP
#define BAYLIB_BAYESIAN_UTILS_HPP


#include <baylib/network/bayesian_network.hpp>
#include <unordered_set>
#include <baylib/probability/condition_factory.hpp>

/**
 * this file contains some useful algorithms
 * for bayesian network inference
 *  ! graph rank function for vectorization ordering
 *  ! graph rank direct ordering
 *  ! markov-blanket
 */

namespace  bn{
    /**
    * Applies ranking function to the DAG representing the
    * bayesian network
    * @tparam Probability: the type expressing the probability
    * @return              map containing node-rank as key-value couple
    */
    template<typename Probability>
    std::map<bn::vertex<Probability>, int> graph_rank(const std::shared_ptr<bn::bayesian_network<Probability>> &bn){
        auto ranks = std::map<bn::vertex<Probability>, int>{};
        auto vertices = bn->variables();
        std::vector<bn::vertex<Probability>> roots;

        // fill nodes map with 0s
        for(auto & v : vertices) {
            ranks[v.id()] = 0;
            if(bn->is_root(v.name())) roots.push_back(v.id());
        }

        if(roots.empty())
            throw std::runtime_error("No root nodes found in graph.");

        while(!roots.empty()) {
            bn::vertex<Probability> curr_node = roots.back();
            roots.pop_back();

            for (auto v : vertices) {
                auto vid = v.id();
                if (!bn->has_dependency(curr_node, vid)) continue;

                if (ranks[curr_node] + 1 > ranks[vid]) {
                    ranks[vid] = ranks[curr_node] + 1;
                    roots.push_back(vid);
                }
            }
        }

        return ranks;
    }

    /**
     *
     * @tparam Probability: the type expressing the probability
     * @param  bn         : bayesian network shared pointer
     * @return            : vector containing variables sorted by rank
     */
    template <typename Probability>
    std::vector<bn::vertex<Probability>> rank_order (const std::shared_ptr<bn::bayesian_network<Probability>> &bn){
       using pair_t = std::pair<bn::vertex<Probability>, int>;

       auto tmp = std::vector<pair_t>{};
       auto rank_map = graph_rank(bn);
       auto order = std::vector<bn::vertex<Probability>>{};

       tmp.reserve(rank_map.size());
       std::copy(rank_map.begin(), rank_map.end(), tmp.begin());

       std::sort(tmp.begin(), tmp.end(), [](const pair_t &a, const pair_t &b){
           return a.second < b.second;
       });

       std::transform(tmp.begin(), tmp.end(), std::back_inserter(order), [](auto p){
           return p.second;
       });

       return order;
    }

    /**
     * Computes the Markov blanket reduction
     * given the bayesian network and a node
     * @tparam Probability : the type expressing the probability
     * @param bn           : bayesian network shared pointer
     * @param rv           : random variable node
     * @return             : unordered set containing the Markov blanket
     */
    template <typename Probability>
    std::unordered_set<bn::random_variable<Probability>> markov_blanket
     (  const std::shared_ptr<bn::bayesian_network<Probability>> &bn,
        const bn::random_variable<Probability> &rv
     ) {
        auto marblank = std::unordered_set<bn::random_variable<Probability>>{};
        for(auto & pv : bn->parents_of(rv.name))
            marblank.insert(pv);

        for(auto & cv : bn->children_of(rv.name)){
            marblank.insert(cv);
            for(auto pv : bn->parents_of(cv))
                if(cv.id != rv.id)
                    marblank.insert(pv);
        }

        return marblank;
    }

    /**
     * check whether the given node's condition probability
     * table is properly filled (proper number of entries
     * and sum(row_i) == 1.0 with a tolerance of 1.0e-5)
     * @tparam Probability : the type expressing the probability
     * @param cpt_owner    : the node the cpt belongs to
     * @return             : true if filled out, false otherwise
     */
    template <typename Probability>
    bool cpt_filled_out(bn::random_variable<Probability> &cpt_owner)
    {
        bn::condition_factory factory(cpt_owner);
        if(factory.number_of_combinations() != cpt_owner.table().size())
            return false;

        auto &cpt = cpt_owner.table();
        do {
            auto cond = factory.get();
            if(!cpt.has_entry_for(cond))
                return false;
        } while(factory.has_next());

        for(auto & row : cpt) {
            Probability sum = std::accumulate(row.begin(), row.end(), 0.0);
            if (abs(sum - 1.0) > 1.0e-5) {
                return false;
            }
        }

        return true;
    }

} // namespace bn

#endif //BAYLIB_BAYESIAN_UTILS_HPP
