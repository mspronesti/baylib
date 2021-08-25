#ifndef BAYLIB_NET_PARSER_HPP
#define BAYLIB_NET_PARSER_HPP
#include <string>
#include <fstream>
#include <sstream>

#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/cpt.hpp>
#include "rapidxml/rapidxml.hpp"

#include <baylib/probability/condition_factory.hpp>

using namespace rapidxml;

//reader class for bayesian network stored in .xdsl files
//the template parameter is used to define the precision of the probability red from the file

namespace bn {
    template<typename Probability>
    class net_parser {
    public:
        net_parser() = default;

        //loads the bayesian network from the given file
        bn::bayesian_network<Probability> load_from_xdsl(const std::string &fileName) {

            bn::bayesian_network<Probability> net;
            auto doc = std::make_shared<xml_document<>>();
            std::ifstream inputFile(fileName);

            BAYLIB_ASSERT(inputFile,
                    "file " << fileName
                    << " does not exist",
                    std::runtime_error)

            auto buffer = std::make_shared<std::stringstream>();
            *buffer << inputFile.rdbuf();
            inputFile.close();
            std::string content(buffer->str());
            doc->parse<0>(&content[0]);
            xml_node<> *pRoot = doc->first_node();
            xml_node<> *pNodes = pRoot->first_node("nodes");

            //reading all variables in file
            for (xml_node<> *pNode = pNodes->first_node("cpt"); pNode; pNode = pNode->next_sibling()) {
                xml_attribute<> *attr = pNode->first_attribute("id");
                std::string varname;
                std::vector<std::string> state_names;
                std::vector<std::string> parents;
                std::vector<Probability> probDistribution;
                std::vector<std::string> resultingStates;

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

                // Build bayesian_network
                net.add_variable(varname, state_names);
                for (auto parent: parents)
                    net.add_dependency(parent, varname);



                // fill CPTs

                std::reverse(parents.begin(), parents.end());
                bn::condition_factory cf(net[varname], parents);
                unsigned int i = 0;
                do {
                    auto cond = cf.get();
                    for (int j = 0; j < state_names.size(); j++)
                        net.set_variable_probability(varname, j, cond, probDistribution[i * state_names.size()  + j]);
                    ++i;
                } while (cf.has_next());

            }
            return net;
        }

    private:
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
#endif //BAYLIB_NET_PARSER_HPP
