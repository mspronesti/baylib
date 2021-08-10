#ifndef BAYESIAN_INFERRER_NET_PARSER_HPP
#define BAYESIAN_INFERRER_NET_PARSER_HPP
#include <string>
#include <fstream>
#include <sstream>

#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/cpt.hpp>
#include "rapidxml/rapidxml.hpp"

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

            if(!inputFile)
                throw std::runtime_error("File does not exists");

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
                std::string varName;
                std::vector<std::string> stateNames;
                std::vector<std::string> parents;
                std::vector<Probability> probDistribution;
                std::vector<std::string> resultingStates;

                //reading variable name
                varName = attr->value();

                //reading all properties of the variable
                for (xml_node<> *pStates = pNode->first_node("state"); pStates; pStates = pStates->next_sibling()) {
                    std::string name(pStates->name());

                    //reading variable's states
                    if (name == "state") {
                        attr = pStates->first_attribute("id");
                        if (attr != nullptr)
                            stateNames.emplace_back(attr->value());
                    }
                    else if (name == "resultingstates")
                        resultingStates.emplace_back(pStates->value());
                    else if (name == "probabilities")
                        probDistribution = split<Probability>(pStates->value(), [](const std::string &t){return static_cast<Probability>(std::stod(t));});
                    else if (name == "parents")
                        parents = split<std::string>(pStates->value(), [](const std::string &t){return t;});
                }

                // Build bayesian_network
                net.add_variable(varName, stateNames);
                for (auto parent: parents)
                    net.add_dependency(parent, varName);

                for (int i = 0; i < probDistribution.size()/stateNames.size(); ++i)
                    for(int j = 0; j < stateNames.size(); ++j)
                        net.set_variable_probability(varName, j, i, probDistribution[i * stateNames.size() + j]);
            }
            return net;


        }

    private:
        template<class T>
        std::vector<T> split(const std::string &text, std::function<T(const std::string&)> mapper, const std::string& delimiter = " "){
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
#endif //BAYESIAN_INFERRER_NET_PARSER_HPP
