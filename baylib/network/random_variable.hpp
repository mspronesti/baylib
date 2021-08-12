//
// Created by elle on 10/08/21.
//

#ifndef BAYLIB_RANDOM_VARIABLE_HPP
#define BAYLIB_RANDOM_VARIABLE_HPP

#include <baylib/network/bayesian_network.hpp>

#include <boost/range/adaptor/map.hpp>
#include <boost/range/algorithm/copy.hpp>

namespace  bn {
    // forward declaration
    template<typename Probability>
    class bayesian_network;

    template<typename Probability>
    class random_variable {

        std::string _name;
        bn::cow::cpt<Probability> cpt;
        std::vector <std::string> _states;
        unsigned long _id{};

        // stored to avoid passing the graph over and over
        // when not needed
        std::map<std::string, int> parents_states;

        friend class bn::bayesian_network<Probability>;

    public:
        random_variable() = default;

        random_variable(std::string name, const std::vector <std::string> &states)
                : _name(std::move(name)), cpt(states.size()), _states(states) {}

        bool has_state(const std::string &state_name) const {
            return std::any_of(_states.begin(), _states.end(),
                               [state_name](std::string state) { return state_name == state; });
        }

        void set_probability(
           bn::state_t state_value,
           const bn::condition &cond,
           Probability p
        )
        {
            cpt.set_probability(cond, state_value, p);
        }

        std::string name() const {
            return _name;
        }

        std::vector <std::string> states() const {
            return _states;
        }

        bn::cow::cpt<Probability> &table() {
            return cpt;
        }

        unsigned long id() const {
            return _id;
        }

        int parent_states_size(const std::string &name) {
           auto it = parents_states.find(name);
           if(it != parents_states.end())
               return it->second;

           return -1;  // doesn't throw on purpose
        }

        std::vector<std::string> parents_names () {
          auto pnames = std::vector<std::string>{};
          boost::copy(parents_states | boost::adaptors::map_keys,
                                       std::back_inserter(pnames));
          return pnames;
        }

    };
} // namespace bn


#endif //BAYLIB_RANDOM_VARIABLE_HPP
