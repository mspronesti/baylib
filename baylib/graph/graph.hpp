//
// Created by elle on 01/08/21.
//

#ifndef BAYLIB_GRAPH_HPP
#define BAYLIB_GRAPH_HPP

#include <boost/graph/adjacency_list.hpp>
#include <baylib/probability/cpt.hpp>

namespace bn {
    template <typename Vertex>
    using graph = boost::adjacency_list<boost::listS, // edge container
                                        boost::vecS, // vertex container
                                        boost::bidirectionalS, // graph type
                                        Vertex // vertex type
                                        >;
    template <typename Vertex>
    using vertex = typename graph<Vertex>::vertex_descriptor;

} // namespace bn

#endif //BAYLIB_GRAPH_HPP
