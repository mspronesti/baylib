//
// Created by paolo on 04/07/2021.
//

#include "BayesianNet.h"
#include "../pugixml/pugixml.hpp"


using namespace std;
namespace ub = boost::numeric::ublas;
class FileNotFoundException: exception{
};


/// Function to read xdsl format and transform it in a vector of bnodes, the nodes are indipendent and
/// the graph is not generated
/// \param doc xml document handle from the pugi library
/// \param res_list vector where the results will be saved
void readXML(const pugi::xml_document& doc, vector<bnode> &res_list){
    pugi::xml_node node = doc.first_child().child("nodes");
    map<string, int> nevent;
    for (auto cpt:node.children()){
        list<string> event{};
        ub::matrix<double> matr{};
        string name = cpt.first_attribute().value();
        string probs_s = cpt.child("probabilities").child_value();
        string parent_s = cpt.child("parents").child_value();
        list<string> parents{};
        int ncolumns = 0;
        for (auto el: cpt.children("state")) {
            event.emplace_back(el.attribute("id").value());
        }
        size_t pos;
        string token;
        while((pos = parent_s.find(' ')) != string::npos){
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
        int i=0;
        while((pos = probs_s.find(' ')) != string::npos){
            token = probs_s.substr(0, pos);
            probs_s.erase(0, pos+1);
            matr(i%event.size(), i/event.size()) = std::stod(token);
            i++;
        }
        matr(i%event.size(), i/event.size()) = std::stod(probs_s);
        res_list.emplace_back(bnode(name, event, parents,matr));
    }
}

/// Constructor of Bayesian Network from file
/// \param file_name path
BayesianNet::BayesianNet(const char *file_name) {
    pugi::xml_document doc;

    if(!doc.load_file(file_name))
        throw FileNotFoundException();

    readXML(doc, bnode_vec);
    network = Graph(bnode_vec.size());

    for(int i=0; i<bnode_vec.size(); i++)
        name_map[bnode_vec[i].getName()] = i;

    for(int i=0; i<bnode_vec.size(); i++){
        for(const string& parent: bnode_vec[i].getParents()){
            int j = name_map[parent];
            boost::add_edge(j, i, network);
        }
    }

}

