//
// Created by elle on 02/08/21.
//

#ifndef BAYLIB_BAYESIAN_UTILS_HPP
#define BAYLIB_BAYESIAN_UTILS_HPP


#include <baylib/network/bayesian_net.hpp>
#include <baylib/probability/condition_factory.hpp>
#include <baylib/baylib_concepts.hpp>
#include <unordered_set>

//! \file bayesian_utils.hpp
//! \brief Collection of utilities for bayesian_net

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

namespace  baylib{
    /**
    * Applies ranking function to the DAG representing the
    * bayesian network to get the appropriate sampling order
    *
    * @tparam Network_ : the type expressing the random variable
    * @param  bn       : bayesian network
    * @return          : vector containing variables sorted by rank
    */
    template <BNetDerived Network_>
    std::vector<unsigned long> sampling_order (
            const Network_ &bn
    )
    {
        ulong nvars = bn.number_of_variables();

        // initially, the all have rank 0
        auto ranks = std::vector<unsigned long>(nvars, 0);
        auto roots = std::vector<unsigned long>{};

        for(ulong vid = 0; vid < nvars; ++vid)
            if(bn.is_root(vid))
                roots.push_back(vid);

        BAYLIB_ASSERT(!roots.empty(),
                      "No root vertices found in graph",
                      std::runtime_error)

        while(!roots.empty()) {
            unsigned long curr_node = roots.back();
            roots.pop_back();

            for(ulong vid = 0; vid < nvars; ++vid) {
                if (!bn.has_dependency(curr_node, vid)) continue;

                if (ranks[curr_node] + 1 > ranks[vid]) {
                    ranks[vid] = ranks[curr_node] + 1;
                    roots.push_back(vid);
                }
            }
        }

        auto order = std::vector<unsigned long>(nvars);
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&ranks](auto &a, auto&b){
            return ranks[a] < ranks[b];
        });

        return order;
    }

    /**
     * Computes the Markov blanket reduction
     * given the bayesian network and a node
     * @tparam Variable_ : the type of bayesian network
     * @param bn        : bayesian network
     * @param vid       : random variable id
     * @return          : unordered set containing the Markov blanket
     */
    template <BNetDerived Network_>
    std::unordered_set<unsigned long> markov_blanket (
         const Network_ &bn,
         unsigned long vid
     )
     {
        auto marblank = std::unordered_set<unsigned long>{};
        for(auto & pv : bn.parents_of(vid))
            marblank.insert(pv);

        for(auto & v : bn.children_of(vid)){
            marblank.insert(vid);
            for(auto pv : bn.parents_of(v))
                if(v != vid)
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
    template <BNetDerived Network_>
    bool cpt_filled_out(
         const Network_ &bn,
         const unsigned long cpt_owner
    )
    {
        baylib::condition_factory factory(bn, cpt_owner);
        const auto &cpt = bn[cpt_owner].table();

        if(factory.number_of_combinations() != cpt.size())
            return false;

        do {
            auto cond = factory.get();
            if(!cpt.has_entry_for(cond))
                return false;
        } while(factory.has_next());

        for(auto & row : cpt) {
            auto sum = std::accumulate(row.begin(), row.end(), 0.0);
            if (abs(sum - 1.0) > 1.0e-5) {
                return false;
            }
        }

        return true;
    }

    /**
     * Utility to reset all evidences in the given bayesian network
     * @tparam Variable_  : the type expressing the random variable
     * @param bn            : the Bayesian network model
     */
    template <BNetDerived Network_>
    void clear_network_evidences(Network_ &bn)
    {
        std::for_each(bn.begin(), bn.end(), [](auto & var){
            if(var.is_evidence())
                var.clear_evidence();
        });
    }

    /**
     * Utility that returns the list of nodes that are ancestors of evidence including evidences themselves,
     * the nodes are returned in topological order
     * @tparam Variable  : the type expressing the random variable
     * @param bn         : the Bayesian network model
     * @return           : vector of nodes
     */
    template <BNetDerived Network_>
    std::vector<ulong> ancestors_of_evidence(const Network_ &bn){
        std::vector<bool> ancestor(bn.number_of_variables(), false);
        std::function<void(ulong)> mark_ancestors;
        mark_ancestors = [&bn, &ancestor, &mark_ancestors](ulong v_id){
            ancestor[v_id] = true;
            for (ulong p_id: bn.parents_of(v_id)) {
                if(ancestor[p_id])
                    continue;
                ancestor[p_id] = true;
                mark_ancestors(p_id);
            }
        };
        for (uint i = 0; i < bn.number_of_variables(); ++i) {
            if(bn[i].is_evidence())
                mark_ancestors(i);
        }
        std::vector<ulong> ordered_result;
        for (auto vertex: sampling_order(bn)) {
            if(ancestor[vertex])
                ordered_result.emplace_back(vertex);
        }
        return ordered_result;
    }

    /**
     * Function to check if network has any evidence node
     * @tparam Network_ : type of network
     * @param bn        : network
     * @return          : true if any evidence node was found, false if there are no evidence nodes
     */
    template <BNetDerived Network_>
    bool evidence_presence(const Network_ &bn){
        return std::any_of(bn.begin(), bn.end(), [](auto var){return var.is_evidence();});
    }


} // namespace baylib

#endif //BAYLIB_BAYESIAN_UTILS_HPP
