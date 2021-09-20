//
// Created by elle on 01/08/21.
//

#ifndef BAYLIB_GRAPH_HPP
#define BAYLIB_GRAPH_HPP

#include <boost/graph/adjacency_list.hpp>
#include <baylib/probability/cpt.hpp>

#include <boost/range/adaptors.hpp>

//! \file graph.hpp
//! \brief


namespace bn {
    template <typename Vertex>
    using graph = boost::adjacency_list<boost::listS, // edge container
                                        boost::vecS, // vertex container
                                        boost::bidirectionalS, // graph type
                                        Vertex // vertex type
                                        >;
    template <typename Vertex>
    using vertex = typename graph<Vertex>::vertex_descriptor;

    /**
     * retrieves bundled properties (custom vertices) range
     * from the given boost graph transforming boost "vertices"
     * iterator
     * Useful to overload iterators in bn::bayesian_network<Probability>
     * @tparam Graph : the boost graph descriptor
     * @param _g     : the boost graph const reference
     * @return       : bundled properties range
     */
    template <typename Vertex>
    auto bundles(graph<Vertex> & _g)  {
        using boost::adaptors::transformed;
        auto accessor = [map = get(boost::vertex_bundle, _g)](auto vid) -> auto& {
            return map[vid];
        };

        return boost::vertices(_g) | transformed(accessor);
    }


} // namespace bn

#endif //BAYLIB_GRAPH_HPP
