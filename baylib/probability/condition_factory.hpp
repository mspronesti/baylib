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
         *                 needed you can pass your vector,
         *                 otherwise it employs the default vector taken from rv
         */
        explicit condition_factory
                (
                        const bn::random_variable<Probability> &rv,
                        const std::vector<std::string>& parents = {}
                )
                : var(rv)
                , condition_index(0)
                , ncombinations(1)
                , _parents(parents)
        {
            if(parents.empty()) {
                _parents = var.parents_names();
            }
            else{
                auto tmp = std::vector<std::string>(parents);
                // sort takes O(NlogN) while checking permutations
                // checks O(N^2)
                std::sort(tmp.begin(), tmp.end());
                BAYLIB_ASSERT(tmp == rv.parents_names(),
                              "provided parents don't match "
                              "random variable " << rv.name() << " parents",
                              std::runtime_error)
            }

            for (auto &parent : _parents) {
                ncombinations *= var.parent_states_size(parent);
            }
            produce_condition();
        }

        /**
         * computes next condition (if any)
         * employing an "hardware-like" strategy
         * @return true if new condition
         *         false otherwise
         */
        bool has_next() {
            if (++condition_index >= ncombinations)
                return false;

            produce_condition();
            return true;
        }

        /**
         * retrieves the number of total combinations
         * for the given node, i.e. the number of rows
         * in his conditional probability table (CPT)
         * @return combinations number
         */
        std::uint64_t combinations_number () const {
            return ncombinations;
        }

        /**
         * retrieves current condition
         * @return condition
         */
        bn::condition get() {
            return c;
        }

    private:
        /**
         * produces a new condition employing
         * the following formula
         *   f(i,j) = i / prod_{k=0}^{j-1} (ck) mod cj
         *
         * where i  = combination index
         *       j  = random variable index
         *       cj = j-th variable cardinality (number of parents)
         *       prod_{k=0}^{j-1} is the the product for k = {0, 1, ..., j-1}
         */
        void produce_condition() {
            std::uint64_t cum_card = 1;
            for (auto name : _parents) {
                c.add(name, condition_index / cum_card % var.parent_states_size(name));
                cum_card *= var.parent_states_size(name);
            }
        }

        bn::condition c;
        bn::random_variable<Probability> var;
        std::vector<std::string> _parents;
        std::uint64_t condition_index;
        std::uint64_t ncombinations;
    };
}

#endif //BAYLIB_CONDITION_FACTORY_HPP
