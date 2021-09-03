//
// Created by elle on 02/08/21.
//

#ifndef BAYLIB_BAYESIAN_UTILS_HPP
#define BAYLIB_BAYESIAN_UTILS_HPP


#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/condition_factory.hpp>

#include <unordered_set>
#include <mutex>
#include <shared_mutex>

/**
 * this file contains some useful algorithms
 * for bayesian network inference
 *  ! graph rank function for vectorization ordering
 *  ! markov-blanket
 *  ! CPT filled out checker
 *
 *  Other utils might be added in case of future support
 *  of exact inference algorithms
 */

namespace  bn{
    /**
    * Applies ranking function to the DAG representing the
    * bayesian network to get the appropriate sampling order
    *
    * @tparam Probability: the type expressing the probability
    * @param  bn         : bayesian network
    * @return            : vector containing variables sorted by rank
    */
    template <typename Probability>
    std::vector<bn::vertex<Probability>> sampling_order(const bn::bayesian_network<Probability> &bn){
        using vertex_t = bn::vertex<Probability>;
        ulong nvars = bn.number_of_variables();

        // initially, the all have rank 0
        auto ranks = std::vector<vertex_t>(nvars, 0);
        auto roots = std::vector<vertex_t>{};

        for(ulong vid = 0; vid < nvars; ++vid)
            if(bn.is_root(vid))
                roots.push_back(vid);

        BAYLIB_ASSERT(!roots.empty(),
                      "No root vertices found in graph",
                      std::runtime_error)

        while(!roots.empty()) {
            vertex_t curr_node = roots.back();
            roots.pop_back();

            for(ulong vid = 0; vid < nvars; ++vid) {
                if (!bn.has_dependency(curr_node, vid)) continue;

                if (ranks[curr_node] + 1 > ranks[vid]) {
                    ranks[vid] = ranks[curr_node] + 1;
                    roots.push_back(vid);
                }
            }
        }

        auto order = std::vector<vertex_t>(nvars);
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&ranks](auto &a, auto&b){
            return ranks[a] < ranks[b];
        });

        return order;
    }

    /**
     * Computes the Markov blanket reduction
     * given the bayesian network and a node
     * @tparam Probability : the type expressing the probability
     * @param bn           : bayesian network
     * @param rv           : random variable node
     * @return             : unordered set containing the Markov blanket
     */
    template <typename Probability>
    std::unordered_set<bn::vertex<Probability>> markov_blanket
     (  const bn::bayesian_network<Probability> &bn,
        const bn::random_variable<Probability> &rv
     ) {
        auto rv_id = rv.id();
        auto marblank = std::unordered_set<bn::vertex<Probability>>{};
        for(auto & pv : bn.parents_of(rv_id))
            marblank.insert(pv);

        for(auto & vid : bn.children_of(rv_id)){
            marblank.insert(vid);
            for(auto pv : bn.parents_of(vid))
                if(vid != rv_id)
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
    bool cpt_filled_out(const bn::random_variable<Probability> &cpt_owner)
    {
        bn::condition_factory factory(cpt_owner);
        const auto &cpt = cpt_owner.table();

        if(factory.number_of_combinations() != cpt.size())
            return false;

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

    /**
     * Utility to reset all evidences in the given bayesian network
     * @tparam Probability  : the type expressing probability
     * @param bn            : the Bayesian network model
     */
    template <typename Probability>
    void reset_network_evidences(const bn::bayesian_network<Probability> &bn)
    {
        std::for_each(bn.begin(), bn.end(), [](auto & var){
            if(var.is_evidence())
                var.reset_evidence();
        });
    }

} // namespace bn

#endif //BAYLIB_BAYESIAN_UTILS_HPP
