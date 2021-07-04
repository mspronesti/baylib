//
// Created by paolo on 04/07/2021.
//
#include <fstream>
#include <iostream>
#include "pugixml/pugixml.hpp"

using namespace pugi;

int main(){
    xml_document doc;
    if(!doc.load_file("../xml_files/Coma.xdsl")){
        std::cout << "error\n";
        return -1;
    }
    xml_node node = doc.first_child().first_child();
    //std::cout << node.name() << " : " << node.value() << '\n';
    for (auto el:node.children()) {
        int i=0;
        for (auto el2: el.children("state")) {
            std::cout << el2.attribute("id").name() << " : " << el2.attribute("id").value()<< '\n';
            i++;
        }
        std::cout << el.child("probabilities").name() << " : " << el.child("probabilities").child_value() << '\n';
    }
    /*
    for(; tool; tool = tool.next_sibling("Success")){
        for(auto r: tool.attributes()){
            std::cout << r << "\n";
        }
    }*/
}