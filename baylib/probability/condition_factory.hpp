//
// Created by elle on 11/08/21.
//

#ifndef BAYLIB_CONDITION_FACTORY_HPP
#define BAYLIB_CONDITION_FACTORY_HPP

#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/condition.hpp>


namespace  bn {
    template<typename Probability>
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
             const bn::random_variable<Probability> &rv,
             const std::vector<std::string>& parents = {}
        )
        : var(rv)
        , condition_index(0)
        , ncombinations(1)
        , _parents(parents)
        {
            if(parents.empty())
                _parents = var.parents_names();

            // load first condition and compute the number
            // of combinations
            for (auto &parent : _parents) {
                // parent_states_number throws if invalid
                // parent, hence no extra check needed
                c.add(parent, condition_index / ncombinations % var.parent_states_number(parent));
                ncombinations *= var.parent_states_number(parent);
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
            for (auto name : _parents) {
                c.add(name, condition_index / cum_card % var.parent_states_number(name));
                cum_card *= var.parent_states_number(name);
            }
            return true;
        }

        /**
         * retrieves the number of total combinations
         * for the given node, i.e. the number of rows
         * in his conditional probability table (CPT)
         * @return combinations number
         */
        std::uint64_t number_of_combinations() const {
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
        bn::random_variable<Probability> var;
        std::vector<std::string> _parents;
        std::uint64_t condition_index;
        std::uint64_t ncombinations;
    };
}

#endif //BAYLIB_CONDITION_FACTORY_HPP