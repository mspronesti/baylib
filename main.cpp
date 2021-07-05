//
// Created by paolo on 04/07/2021.
//
#include <fstream>
#include <iostream>
#include "pugixml/pugixml.hpp"
#include "graph/BayesianNet.h"

using namespace pugi;

int main(){
    xml_document doc;
    BayesianNet net("../xml_files/Coma.xdsl");
    if(!doc.load_file("../xml_files/Coma.xdsl")){
        std::cout << "error\n";
        return -1;
    }
    xml_node node = doc.first_child().first_child();
    std::cout << node.name() << " : " << node.value() << '\n';
    for (auto el:node.children()) {
        int i=0;
        std::cout << el.first_attribute().name() << " : " << el.first_attribute().value() << '\n';
        for (auto el2: el.children("state")) {
            std::cout << el2.attribute("id").name() << " : " << el2.attribute("id").value()<< '\n';
            i++;
        }
        std::cout << el.child("probabilities").name() << " : " << el.child("probabilities").child_value() << '\n';
    }
}