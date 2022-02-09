//
// Created by paolo on 15/09/21.
//

#ifndef BAYLIB_GPU_UTILS_HPP
#define BAYLIB_GPU_UTILS_HPP

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <utility>
#include <baylib/network/bayesian_net.hpp>
#include <baylib/tools/gpu/cuda_utils.cuh>


/**
 * @file gpu_utils.hpp
 * @brief utils for using boost::compute
 */

#define MEMORY_SLACK 80

namespace baylib {
    /**
     * Container for gpu vectors with built in auto release of the memory after set number of uses
     */
    struct bcvec {

        bcvec() = default;

        bcvec(uint dimension, uint cardinality, boost::compute::context &context) :
                state(dimension, context), cardinality(cardinality) {}

        bcvec(boost::compute::vector<int> state, uint cardinality, uint uses) :
                state(std::move(state)), cardinality(cardinality), isevidence(false), evidence_state(0),
                uses(uses == 0 ? 1 : uses) {}

        bcvec(uint evidence_state, uint cardinality) :
                isevidence(true), cardinality(cardinality), evidence_state(evidence_state), uses(0) {}

        void add_use() {
            uses--;
            if (!uses)
                state.clear();
        }

    private:
        uint uses{};

    public:
        boost::compute::vector<int> state;
        uint cardinality{};
        bool isevidence{};
        uint evidence_state{};

    };

    /**
     * flatten a cpt into vector preserving the condition order given by the network
     * @tparam probability_type     : type of the cpt
     * @tparam Network_             : network type
     * @param bn                    : network
     * @param v_id                  : vertex id from which the cpt comes from
     * @return                      : unrolled cpt
     */
    template<Arithmetic probability_type, BNetDerived Network_>
    std::vector<probability_type> flatten_cpt(const Network_ &bn, ushort v_id) {
        auto factory = baylib::condition_factory(bn, v_id, bn.parents_of(v_id));
        std::vector<probability_type> flat_cpt{};
        do {
            auto temp = bn[v_id].table()[factory.get()];
            flat_cpt.insert(flat_cpt.end(), temp.begin(), temp.end());
        } while (factory.has_next());
        return flat_cpt;
    }

    template<typename probability_type, BNetDerived Network_>
    marginal_distribution <probability_type> reshape_marginal(const Network_ &bn,
                                                              const std::vector<ulong> &v_order,
                                                              const std::vector<uint> &flat_marginal) {
        marginal_distribution<probability_type> result(bn.begin(), bn.end());
        uint marginal_ix = 0;
        for (ulong i: v_order) {
            for (int j = 0; j < bn[i].states().size(); ++j) {
                result[i][j] = static_cast<probability_type>(flat_marginal[marginal_ix]);
                marginal_ix++;
            }
        }
        return result;
    }

    /**
     * upload bayesian network to cuda memory
     * @tparam probability_type : probability type
     * @tparam Network_         : network type
     * @param bn                : network
     * @return                  : cuda_graph
     */
    template<typename probability_type, BNetDerived Network_>
    cuda_graph<probability_type> make_cuda_graph(const Network_ &bn) {
        auto graph = cuda_graph<probability_type>(bn.number_of_variables());
        for (ushort i = 0; i < bn.number_of_variables(); i++) {

            graph.add_variable(i, baylib::flatten_cpt<probability_type>(bn, i),
                               bn[i].table().number_of_states(),
                               bn.parents_of(i).size(),
                               bn[i].table().number_of_conditions());
            graph.add_dependencies(i, bn.parents_of(i));
        }
        return graph;
    };
}

#endif //BAYLIB_GPU_UTILS_HPP