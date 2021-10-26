#ifndef BAYLIB_SMILE_UTILS_HPP
#define BAYLIB_SMILE_UTILS_HPP

#include <string>
#include <fstream>
#include <sstream>

#include <baylib/network/bayesian_net.hpp>
#include <baylib/probability/cpt.hpp>
#include "rapidxml/rapidxml.hpp"

#include <baylib/probability/condition_factory.hpp>
#include <baylib/baylib_concepts.hpp>

using namespace rapidxml;

//! \file smile_utils.hpp
//! \brief utilities to support the xdsl format from SMILE library

namespace baylib {
    /**
     * This class is used to support xdsl additional features associated
     * to a random variable, inheriting from random_variable.
     * Offers to possibility to name the variable and its states.
     * @tparam Probability_ : the type expressing the probability
     *                        must be arithmetic
     */
    template <Arithmetic Probability_ = double>
    class named_random_variable : public baylib::random_variable<Probability_> {
    public:
        named_random_variable() = default;

        named_random_variable (
                std::string name,
                const std::vector <std::string> &states
        )
        : random_variable<Probability_>(states.size())
        , _name(std::move(name))
        , _states(states)
        { }

        /**
        * @return vector of state names
        */
        std::vector <std::string> states() const {
            return _states;
        }

        /**
         *  Retrieves the name of the state
         *  identified by "s" as an index
         * @param s: number of state
         * @return name of s-th state
         */
        std::string state(unsigned long s) const {
            BAYLIB_ASSERT(s < _states.size(),
                          "random variable " << _name
                          << " has " << _states.size()
                          << " states, but " << s
                          << "-th was requested",
                          std::runtime_error)
            return _states[s];
        }

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
        * @return name of variable
        */
        std::string name() const {
            return _name;
        }

    private:
        std::string _name;
        std::vector <std::string> _states;
    };

  /**
  * This methods builds a {name: id} map scanning a bayesian_net
  * whose nodes are named_random_variables
  * @tparam Variable_
  * @param bn : bayesian network of type Network_
  * @return
  */
    template <RVarDerived Variable_>
#ifdef __concepts_supported
    requires std::is_same_v <
                            Variable_,
                            named_random_variable<typename Variable_::probability_type>
                            >
#endif
    std::map<std::string, unsigned long> make_name_map (
            const baylib::bayesian_net<Variable_> & bn
    )
    {
        auto name_map = std::map<std::string, unsigned long>{};
        std::for_each(bn.begin(), bn.end(), [&name_map](const auto & var){
            name_map[var.name()] = var.id();
        });

        return name_map;
    }

    /**
     * This class models an xml parser for the xdsl format, used
     * to support the compatibility with the SMILE library
     * @tparam Probability_ : the type expressing the probability
     *                        must be arithmetic
     */
    template<Arithmetic Probability_ = double>
    class xdsl_parser {
        typedef baylib::bayesian_net<named_random_variable<Probability_>> named_bayesian_network;
    public:
        xdsl_parser() = default;

        /**
         * Create a bayesian_net starting from an xdsl file, the format is the same as specified in the
         * smile library, if specified file can't be found an exception is thrown
         * @param file_name : file name
         * @return         : bayesian network
         */
        named_bayesian_network deserialize (
             const std::string & file_name
        )
        {
            named_bayesian_network bn;
            auto doc = std::make_shared<xml_document<>>();
            std::ifstream input_file(file_name);

            BAYLIB_ASSERT(input_file,
                          "file " << file_name << " was"
                          " not found in current path",
                          std::runtime_error)

            auto buffer = std::make_shared<std::stringstream>();
            *buffer << input_file.rdbuf();
            input_file.close();

            std::string content(buffer->str());
            doc->parse<0>(&content[0]);
            xml_node<> *pRoot = doc->first_node();
            xml_node<> *pNodes = pRoot->first_node("nodes");
            std::map<std::string, ulong> name_map{};

            //reading all variables in file
            for (xml_node<> *pNode = pNodes->first_node("cpt"); pNode; pNode = pNode->next_sibling()) {
                xml_attribute<> *attr = pNode->first_attribute("id");
                std::string varname;
                std::vector<std::string> state_names, parents, resultingStates;
                std::vector<Probability_> probDistribution;

                //reading variable name
                varname = attr->value();

                //reading all properties of the variable
                for (xml_node<> *pStates = pNode->first_node("state"); pStates; pStates = pStates->next_sibling()) {
                    std::string name(pStates->name());

                    //reading variable's states
                    if (name == "state") {
                        attr = pStates->first_attribute("id");
                        if (attr != nullptr)
                            state_names.emplace_back(attr->value());
                    }
                    else if (name == "resultingstates") {
                        for (auto state:split<std::string>(pStates->value(), [](const std::string &t){return t;})) {
                            int ix = std::find(state_names.begin(), state_names.end(), state) - state_names.begin();
                            for (int i = 0; i < state_names.size(); ++i)
                                probDistribution.emplace_back(i == ix ? 1. : 0.);
                        }
                    }
                    else if (name == "probabilities")
                        probDistribution = split<Probability_>(pStates->value(), [](const std::string &t){return static_cast<Probability_>(std::stod(t));});
                    else if (name == "parents")
                        parents = split<std::string>(pStates->value(), [](const std::string &t){return t;});
                }

                // Build the bayesian_net
                ulong var_id = bn.add_variable(varname, state_names);
                name_map[varname] = var_id;
                std::vector<ulong> parents_id{};
                for (const auto& parent: parents){
                    bn.add_dependency(name_map[parent], var_id);
                    parents_id.emplace_back(name_map[parent]);
                }

                // fill CPTs
                std::reverse(parents_id.begin(), parents_id.end());
                baylib::condition_factory cf(bn, var_id, parents_id);
                unsigned int i = 0;
                do {
                    auto cond = cf.get();
                    for (int j = 0; j < state_names.size(); j++)
                        bn.set_variable_probability(var_id, j, cond, probDistribution[i * state_names.size() + j]);
                    ++i;
                } while (cf.has_next());

            }
            return bn;
        }

    private:

        /**
         * Utility of the parser for splitting a string and mapping to a specific value
         * @tparam T        : output of mapper
         * @param text      : input string
         * @param mapper    : mapper function
         * @param delimiter : delimiter for splitting the input string
         * @return          : output vector
         */
        template<typename T>
        std::vector<T> split(
                const std::string &text,
                std::function<T(const std::string&)> mapper,
                const std::string& delimiter = " "
        )
        {
            std::vector<T> result;
            unsigned long ix;
            unsigned long start = 0;
            while((ix = text.find(delimiter, start)) != std::string::npos){
                result.emplace_back(mapper(text.substr(start, ix-start)));
                start = ix + delimiter.length();
            }
            result.emplace_back(mapper(text.substr(start, ix-start)));
            return result;
        }
    };


} // namespace baylib
#endif //BAYLIB_SMILE_UTILS_HPP
