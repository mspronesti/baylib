//
// Created by elle on 10/08/21.
//

#ifndef BAYLIB_RANDOM_VARIABLE_HPP
#define BAYLIB_RANDOM_VARIABLE_HPP

#include <baylib/network/bayesian_network.hpp>

#include <boost/range/adaptor/map.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <utility>

namespace  bn {
    // forward declaration
    template<typename Probability>
    class bayesian_network;

    template<typename Probability>
    class random_variable {
    public:
        random_variable() = default;

        random_variable(
            std::string name,
            const std::vector <std::string> &states
        )
        : _name(std::move(name))
        , cpt(states.size())
        , _states(states)
        , _is_evidence(false)
        , _state_value(0)
        { }

        bool has_state(const std::string &state_name) const {
            return std::any_of(_states.begin(), _states.end(),
                               [state_name](const std::string& state) { return state_name == state; });
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

        const  bn::cow::cpt<Probability> &table() const {
            return cpt;
        }

        unsigned long id() const {
            return _id;
        }

        void set_as_evidence(unsigned long value){
            BAYLIB_ASSERT(value < _states.size(),
                          "Invalid value for random"
                          " variable" << _name,
                          std::runtime_error)
            _state_value = value;
            _is_evidence = true;
        }

        void clear_evidence() {
            _is_evidence = false;
        }

        bool is_evidence() const {
            return _is_evidence;
        }

        unsigned long evidence_state() const {
            BAYLIB_ASSERT(_is_evidence,
                          "Random variable " << _name
                          << " is not an evidence",
                          std::logic_error)

            return _state_value;
        }

        struct parents_info_t {
            /**
             * this struct stores the names and the number of
             * states of each parent of the current node (random_variable)
             * to avoid passing the graph over and over when not needed.
             */
        public:
            void add (const std::string &name, ulong nstates) {
                parents_map[name] = nstates;
            }

            void remove(const std::string &name){
                parents_map.erase(name);
            }

            std::vector<std::string> names() {
                auto pnames = std::vector<std::string>{};
                boost::copy(parents_map | boost::adaptors::map_keys,
                            std::back_inserter(pnames));
                return pnames;
            }

            unsigned long num_states_of(const std::string &name){
                auto it = parents_map.find(name);
                BAYLIB_ASSERT( it != parents_map.end(),
                               name << " doesn't represent a"
                                       " valid parent for variable",
                               std::logic_error)

                return it->second;
            }
        private:
            std::map<std::string, unsigned long> parents_map;
        };

        // public because encapsulation is performed inside
        parents_info_t parents_info;
    private:

        friend class bn::bayesian_network<Probability>;

        std::string _name;
        bn::cow::cpt<Probability> cpt;
        std::vector <std::string> _states;
        bool _is_evidence{};
        unsigned long _state_value;
        unsigned long _id{};

    };
} // namespace bn

#endif //BAYLIB_RANDOM_VARIABLE_HPP
