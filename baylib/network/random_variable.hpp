//
// Created by elle on 10/08/21.
//

#ifndef BAYLIB_RANDOM_VARIABLE_HPP
#define BAYLIB_RANDOM_VARIABLE_HPP

#include <baylib/network/bayesian_network.hpp>
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
        /**
         * random variable constructor
         * @param name   : name of the variable
         * @param states : vector of possible states
         */
        explicit random_variable( unsigned long num_states = 2)
        : cpt(num_states)
        , _is_evidence(false)
        , _state_value(0)
        { }


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

        void set_id(unsigned long id) { _id = id; }

        /**
         * @return the number of variable's states
         */
        unsigned long number_of_states() const {
            return cpt.number_of_states();
        }

        /**
         * set the variable as an observed evidence for a specific state, this will alter
         * how the inference process will be executed
         * @param value : evidence state
         */
        void set_as_evidence(unsigned long value){
            BAYLIB_ASSERT(value < number_of_states(),
                          "Invalid value for random"
                          " variable" << _id,
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
                          "Random variable " << _id
                          << " is not an evidence",
                          std::logic_error)

            return _state_value;
        }

    protected:
        template <typename Variable> friend class bayesian_network;

        bn::cow::cpt<Probability> cpt;
        bool _is_evidence;
        unsigned long _state_value;
        unsigned long _id;

    };
} // namespace bn

#endif //BAYLIB_RANDOM_VARIABLE_HPP
