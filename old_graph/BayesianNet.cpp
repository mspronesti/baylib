
#include "BayesianNet.h"
#include "../exceptions/FileNotFoundException.h"
#include "../parser/pugixml/pugixml.hpp"
#include "../parser/parser.h"

namespace ub = boost::numeric::ublas;


/**
 * Constructor of Bayesian Network from file
 * @param file_name
 */
bn::BayesianNet::BayesianNet(const std::string &file_name) {
    pugi::xml_document doc;

    if(!doc.load_file(file_name.c_str()))
        throw FileNotFoundException("Specified path is not valid");

    bnode_vec = readXML(doc);
    network = Graph(bnode_vec.size());

    for(int i = 0; i < bnode_vec.size(); i++)
        name_map[bnode_vec[i].name] = i;

    for(int i = 0; i < bnode_vec.size(); i++)
        for(auto& parent: bnode_vec[i].parents){
            int j = name_map[parent];
            boost::add_edge(j, i, network);
        }
}

