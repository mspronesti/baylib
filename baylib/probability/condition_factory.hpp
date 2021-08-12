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
        explicit condition_factory(const bn::random_variable<Probability> &rv, std::vector<std::string> parents={})
        : var(rv), parents(parents.empty() ? var.parents_names() : parents)
        {
            /*
             auto parents = var.parents_names();

            for (auto &parent : parents) {
                c.add(parent, 0);
            }
             */
            index = 0;
            tot = 1;
            for (auto &parent : parents)
                tot *= var.parent_states_size(parent);
        }

        /**
         * computes next condition (if any)
         * employing an "hardware-like" strategy
         * @return true if new condition
         *         false otherwise
         */
        bool has_next() {
            /*for (auto it = c.rbegin(); it != c.rend(); ++it) {
                auto& key = it->first;
                auto& val = it->second;
                c[key]++;

                if (val >= var.parent_states_size(key))
                    c[key] = 0;
                else
                    return true;
            }*/
            return ++index < tot;
        }


        /**
         * retrieves current condition
         * @return condition
         */
        bn::condition get() {
            bn::condition c;
            uint32_t cum_card = 1;
            for (auto name : parents) {
                c.add(name,index / (cum_card) % (var.parent_states_size(name)));
                cum_card *= var.parent_states_size(name);
            }
            return c;
        }

    private:
        // bn::condition c;
        bn::random_variable<Probability> var;
        std::vector<std::string> parents;
        uint16_t index;
        uint16_t tot;
    };
}

#endif //BAYLIB_CONDITION_FACTORY_HPP
