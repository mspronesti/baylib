#ifndef BAYLIB_XDSL_PARSER_HPP
#define BAYLIB_XDSL_PARSER_HPP

#include <string>
#include <fstream>
#include <sstream>

#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/cpt.hpp>
#include "rapidxml/rapidxml.hpp"

#include <baylib/probability/condition_factory.hpp>

using namespace rapidxml;

//! \file smile_utils.hpp
//! \brief utilities to support the xdsl format from SMILE library

namespace bn {
    template<typename Probability>
    class named_random_variable : public random_variable<Probability> {
    public:
        named_random_variable() = default;

        named_random_variable (
                std::string name,
                const std::vector <std::string> &states
        )
        : random_variable<Probability>(states.size())
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
  * This methods builds a {name: id} map scanning a bayesian_network
  * whose nodes are named_random_variables
  * @tparam Variable
  * @param bn
  * @return
  */
    template <typename Variable>
    std::map<std::string, unsigned long> make_name_map (
            const bn::bayesian_network<Variable> & bn
    )
    {
        auto name_map = std::map<std::string, unsigned long>{};
        std::for_each(bn.begin(), bn.end(), [&name_map](const auto & var){
            name_map[var.name()] = var.id();
        });

        return name_map;
    }


    template<typename Probability>
    class xdsl_parser {
            typedef bn::bayesian_network<named_random_variable<Probability>> __named_bayesian_network;
    public:
        xdsl_parser() = default;

        /**
         * Create a bayesian_network starting from an xdsl file, the format is the same as specified in the
         * smile library, if specified file can't be found an exception is thrown
         * @param file_name : file name
         * @return         : bayesian network
         */
        __named_bayesian_network deserialize (
             const std::string & file_name
        )
        {
            __named_bayesian_network bn;
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
                std::vector<Probability> probDistribution;

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
                        probDistribution = split<Probability>(pStates->value(), [](const std::string &t){return static_cast<Probability>(std::stod(t));});
                    else if (name == "parents")
                        parents = split<std::string>(pStates->value(), [](const std::string &t){return t;});
                }

                // Build the bayesian_network
                ulong var_id = bn.add_variable(varname, state_names);
                name_map[varname] = var_id;
                std::vector<ulong> parents_id{};
                for (const auto& parent: parents){
                    bn.add_dependency(name_map[parent], var_id);
                    parents_id.emplace_back(name_map[parent]);
                }

                // fill CPTs
                std::reverse(parents_id.begin(), parents_id.end());
                bn::condition_factory cf(bn, var_id, parents_id);
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


} // namespace bn
#endif //BAYLIB_XDSL_PARSER_HPP
