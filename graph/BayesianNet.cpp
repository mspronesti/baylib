//
// Created by paolo on 04/07/2021.
//

#include <iostream>
#include "BayesianNet.h"
#include "../pugixml/pugixml.hpp"
#include "bnode.h"
#include <boost/numeric/ublas/matrix.hpp>

using namespace std;
namespace ub = boost::numeric::ublas;
class FileNotFoundException: exception{
};


BayesianNet::BayesianNet(const char *file_name) {
    list<bnode> bnode_lis{};
    pugi::xml_document doc;
    if(!doc.load_file(file_name))
        throw FileNotFoundException();
    pugi::xml_node node = doc.first_child().child("nodes");
    for (auto cpt:node.children()){
        list<string> event{};
        ub::matrix<double> matr{};
        string name = cpt.first_attribute().value();
        string probs_s = cpt.child("probabilities").child_value();
        string parent_s = cpt.child("parents").child_value();
        list<string> parents{};
        for (auto el: cpt.children("state")) {
            event.emplace_back(el.attribute("id").value());
        }
        size_t pos;
        string token;
        while((pos = parent_s.find(' ')) != string::npos){
            token = parent_s.substr(0, pos);
            parents.emplace_back(token);
            parent_s.erase(0, pos + 1);
        }
        if(!parent_s.empty())
            parents.emplace_back(parent_s);
        matr = ub::matrix<double>(event.size(), parent_s.size()+1);
        int i=0;
        while((pos = probs_s.find(' ')) != string::npos){
            token = probs_s.substr(0, pos);
            probs_s.erase(0, pos+1);
            matr(i%event.size(), i/event.size()) = std::stod(probs_s);
            i++;
        }
        matr(i%event.size(), i/event.size()) = std::stod(probs_s);
        bnode_lis.emplace_back(bnode(cpt.value(), event,parents,matr));
    }
    cout << "HI";
}
