//
// Created by elle on 11/08/21.
//

#ifndef BAYLIB_CONDITION_FACTORY_HPP
#define BAYLIB_CONDITION_FACTORY_HPP

#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/condition.hpp>

/**
 * @file condition_factory.hpp
 * @brief Condition factory for generating multiple condition objects sequentially
 */


namespace  bn {
    template<typename Variable_>
    class condition_factory {
        /**
         * this class produces all the combinations
         * for a given random variable of a bayesian
         * network to fill its conditional probability
         * table
         */
    public:
        /**
         * builts a condition factory
         * @param rv     : random variable
         * @param parents: the vector of rv's parents.
         *                 If a specific order of the parents is
         *                 needed (e.g. to parse xdsl) you can pass
         *                 your vector, otherwise it employs the default
         *                 one, taken from rv
         */
        explicit condition_factory(
             const bn::bayesian_network<Variable_> &bn,
             const unsigned long var_id,
             const std::vector<unsigned long>& parents = {}
        )
        : bn(bn)
        , condition_index(0)
        , ncombinations(1)
        , _parents(parents)
        {
            if(parents.empty())
                _parents = bn.parents_of(var_id);

            // load first condition and compute the number
            // of combinations
            for (auto &parent : _parents) {
                // parent_states_number throws if invalid
                // parent, hence no extra check needed
                c.add(parent, condition_index / ncombinations % bn[parent].number_of_states());
                ncombinations *= bn[parent].number_of_states();
            }
        }

        /**
         * sets next condition (if any) employing
         * the following formula:
         *
         *    f(i,j) = i / prod_{k=0}^{j-1} (ck) mod cj
         *
         * where i  = combination index
         *       j  = random variable index
         *       cj = j-th variable cardinality (number of parent states)
         *       prod_{k=0}^{j-1} is the the product for k = {0, 1, ..., j-1}
         *
         * @return true if new condition
         *         false otherwise
         */
        bool has_next() {
            if (++condition_index >= ncombinations)
                return false;

            // next condition
            std::uint64_t cum_card = 1;
            for (auto parent : _parents) {
                c.add(parent, condition_index / cum_card % bn[parent].number_of_states());
                cum_card *= bn[parent].number_of_states();
            }
            return true;
        }

        /**
         * retrieves the number of total combinations
         * for the given node, i.e. the number of rows
         * in his conditional probability table (CPT)
         * @return combinations number
         */
        unsigned long number_of_combinations() const {
            return ncombinations;
        }

        /**
         * retrieves current condition
         * @return condition
         */
        bn::condition get() const {
            return c;
        }

    private:
        bn::condition c;
        const bn::bayesian_network<Variable_> &bn;
        std::vector<unsigned long> _parents;
        unsigned long condition_index;
        unsigned long ncombinations;
    };
}

#endif //BAYLIB_CONDITION_FACTORY_HPP
