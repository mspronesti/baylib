//
// Created by paolo on 04/07/2021.
//
#include <fstream>
#include <iostream>
#include "pugixml/pugixml.hpp"
#include "graph/BayesianNet.h"
#include "test/testFile.hpp"
using namespace pugi;

int main(){
    BayesianNet net("../xml_files/Coma.xdsl");
    testFile();

}