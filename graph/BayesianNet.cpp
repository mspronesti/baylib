
#include "BayesianNet.h"
#include "../pugixml/pugixml.hpp"
#include "../exceptions/FileNotFoundException.h"

namespace ub = boost::numeric::ublas;

/**
 * Function to read xdsl format and transform it in a vector of bnodes,
 * the nodes are indipendent and the graph is not generated
 * @param doc: xml document handled from the pugi library
 * @param res_list: vector to store the results in
 */
void readXML(const pugi::xml_document& doc, std::vector<bnode> &res_list){
    pugi::xml_node node = doc.first_child().child("nodes");
    std::map<std::string, int> nevent;

    for (auto cpt:node.children()){
        std::list<std::string> event{};
        ub::matrix<double> matr{};
        std::string name = cpt.first_attribute().value();
        std::string probs_s = cpt.child("probabilities").child_value();
        std::string parent_s = cpt.child("parents").child_value();
        std::list<std::string> parents;

        int ncolumns = 0;
        for (auto el: cpt.children("state"))
            event.emplace_back(el.attribute("id").value());

        size_t pos;
        std::string token;
        while((pos = parent_s.find(' ')) != std::string::npos){
            token = parent_s.substr(0, pos);
            parents.emplace_back(token);
            parent_s.erase(0, pos + 1);
            ncolumns += nevent[token];
        }

        if(!parent_s.empty()) {
            parents.emplace_back(parent_s);
            ncolumns += nevent[parent_s];
        }

        nevent[name] = (int)event.size();
        ncolumns = ncolumns == 0 ? 1 : ncolumns;
        matr = ub::matrix<double>(event.size(), ncolumns);

        int i = 0;
        while((pos = probs_s.find(' ')) != std::string::npos){
            token = probs_s.substr(0, pos);
            probs_s.erase(0, pos+1);
            matr( i % event.size(), i / event.size()) = std::stod(token);
            i++;
        }

        matr( i % event.size(), i / event.size()) = std::stod(probs_s);
        res_list.emplace_back(bnode(name, event, parents,matr));
    }
}


/**
 * Constructor of Bayesian Network from file
 * @param file_name
 */
BayesianNet::BayesianNet(const std::string &file_name) {
    pugi::xml_document doc;

    if(!doc.load_file(file_name.c_str()))
        throw FileNotFoundException("Specified path is not valid");

    readXML(doc, bnode_vec);
    network = Graph(bnode_vec.size());

    for(int i = 0; i < bnode_vec.size(); i++)
        name_map[bnode_vec[i].name] = i;

    for(int i = 0; i < bnode_vec.size(); i++)
        for(auto& parent: bnode_vec[i].parents){
            int j = name_map[parent];
            boost::add_edge(j, i, network);
        }
}

