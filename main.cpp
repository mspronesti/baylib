//
// Created by paolo on 04/07/2021.
//
#include <fstream>
#include <iostream>
#include "parser/pugixml/pugixml.hpp"
#include "graph/BayesianNet.h"
#include "test/testFile.hpp"

int main(){
    bn::BayesianNet net("../xml_files/Coma.xdsl");
    testFile();
}