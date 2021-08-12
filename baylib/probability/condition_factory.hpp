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
        explicit condition_factory(const bn::random_variable<Probability> &rv)
        : var(rv)
        {
            auto parents = var.parents_names();

            for (auto &parent : parents) {
                c.add(parent, 0);
            }
        }

        /**
         * computes next condition (if any)
         * employing an "hardware-like" strategy
         * @return true if new condition
         *         false otherwise
         */
        bool has_next() {
            for (auto it = c.rbegin(); it != c.rend(); ++it) {
                auto& key = it->first;
                auto& val = it->second;
                c[key]++;

                if (val >= var.parent_states_size(key))
                    c[key] = 0;
                else
                    return true;
            }

            return false;
        }


        /**
         * retrieves current condition
         * @return condition
         */
        const bn::condition& get() const {
            return c;
        }

    private:
        bn::condition c;
        bn::random_variable<Probability> var;
    };
}

#endif //BAYLIB_CONDITION_FACTORY_HPP
