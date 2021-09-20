//
// Created by elle on 10/08/21.
//

#ifndef BAYLIB_RANDOM_VARIABLE_HPP
#define BAYLIB_RANDOM_VARIABLE_HPP

#include <baylib/network/bayesian_network.hpp>

#include <boost/range/adaptor/map.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <utility>

//! \file random_variable.hpp
//! \brief Node class of bayesian_network

namespace  bn {
    // forward declaration
    template<typename Probability>
    class bayesian_network;

    template<typename Probability>
    class random_variable {
    public:

        random_variable() = default;

        /**
         * random variable constructor
         * @param name   : name of the variable
         * @param states : vector of possible states
         */
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

        /**
         * Verify if a specific state is a possible realization of the variable
         * @param state_name : name of the state
         * @return           : true if state_name is a state of var
         */
        bool has_state(const std::string &state_name) const {
            return std::any_of(_states.begin(), _states.end(),
                               [state_name](const std::string& state) { return state_name == state; });
        }

        /**
         * Set a specific value for a cpt entry
         * @param state_value : state name of the entry
         * @param cond        : condition related to the entry
         * @param p           : probability relative to the entry
         */
        void set_probability(
            bn::state_t state_value,
            const bn::condition &cond,
            Probability p
        )
        {
            cpt.set_probability(cond, state_value, p);
        }

        /**
         * @return name of variable
         */
        std::string name() const {
            return _name;
        }

        /**
         * @return vector of state names
         */
        std::vector <std::string> states() const {
            return _states;
        }

        /**
         * @return cpt relative to the variable
         */
        bn::cow::cpt<Probability> &table() {
            return cpt;
        }

        /**
         * @return cpt relative to the variable
         */
        const  bn::cow::cpt<Probability> &table() const {
            return cpt;
        }

        /**
         * @return unique numerical identifier of the variable
         */
        unsigned long id() const {
            return _id;
        }

        /**
         * set the variable as an observed evidence for a specific state, this will alter
         * how the inference process will be executed
         * @param value : evidence state
         */
        void set_as_evidence(unsigned long value){
            BAYLIB_ASSERT(value < _states.size(),
                          "Invalid value for random"
                          " variable" << _name,
                          std::runtime_error)
            _state_value = value;
            _is_evidence = true;
        }

        /**
         * unset the evidence state if it was previously set
         */
        void clear_evidence() {
            _is_evidence = false;
        }

        /**
         * @return true if the node was set as an evidence
         */
        bool is_evidence() const {
            return _is_evidence;
        }

        /**
         * return the evidence state if it was previously set, if no evidence was set
         * an exception is thrown
         * @return numerical identifier of the evidence state
         */
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

            /**
             * add parent's data
             * @param name    : name of parent
             * @param nstates : number of states of parent
             */
            void add (const std::string &name, ulong nstates) {
                parents_map[name] = nstates;
            }

            /**
             * remove parent's dataa
             * @param name : name of parent
             */
            void remove(const std::string &name){
                parents_map.erase(name);
            }

            /**
             * get the vector of parents names
             * @return : vector of names
             */
            std::vector<std::string> names() {
                auto pnames = std::vector<std::string>{};
                boost::copy(parents_map | boost::adaptors::map_keys,
                            std::back_inserter(pnames));
                return pnames;
            }

            /**
             * number of states of a specific parent
             * @param name : name of parent
             * @return     ; number of states of the parent
             */
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
