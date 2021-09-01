//
// Created by elle on 01/08/21.
//

#ifndef BAYLIB_GRAPH_HPP
#define BAYLIB_GRAPH_HPP

#include <boost/graph/adjacency_list.hpp>
#include <baylib/probability/cpt.hpp>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>

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
     * retrieves bundled properties (custom vertices list)
     * from the given boost graph
     * @tparam Graph : the boost graph descriptor
     * @param _g     : the boost graph
     * @return       : bundled properties list
     */
    template <typename Vertex>
    auto bundles(const graph<Vertex> & _g)  {
        using boost::adaptors::transformed;
        auto accessor = [map = get(boost::vertex_bundle, _g)](auto vid) -> auto& {
            return map[vid];
        };

        auto make_view = [=](auto range) {
            return std::vector<Vertex>(
                    boost::begin(range), boost::end(range));
        };

        return make_view(make_iterator_range(boost::vertices(_g) | transformed(accessor)));
    }


} // namespace bn

#endif //BAYLIB_GRAPH_HPP
