//
// Created by paolo on 01/03/22.
//

#ifndef BAYLIB_VECTORIZED_INFERENCE_OPENCL_HPP
#define BAYLIB_VECTORIZED_INFERENCE_OPENCL_HPP

#define CL_TARGET_OPENCL_VERSION 220
#define MEMORY_SLACK .8

#include <baylib/inference/abstract_inference_algorithm.hpp>
#include <baylib/network/bayesian_utils.hpp>
#include <baylib/probability/marginal_distribution.hpp>
#include <baylib/tools/random/random_generator.hpp>
#include <baylib/tools/gpu/gpu_utils.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <future>
#include <baylib/baylib_concepts.hpp>

namespace baylib::inference {

    namespace compute = boost::compute;
    using boost::compute::lambda::_1;
    using boost::compute::lambda::_2;
    /**
         * This class models an approximate inference algorithm
         * vectorized with a GPGPU approach.
         * the method simulate_node samples a node given the results of
         * previous simulations of its parents nodes
         * @tparam Network_  : the type of bayesian network
         */
    template < BNetDerived Network_ >
    class vectorized_inference_algorithm : public inference_algorithm<Network_>
    {
    public:
        typedef Network_ network_type;
        using typename inference_algorithm<Network_>::probability_type;
        using  inference_algorithm<Network_>::bn;

        vectorized_inference_algorithm(
                const Network_ & bn,
                unsigned long n_samples,
                unsigned long memory,
                unsigned int seed = 0,
                const compute::device &device = compute::system::default_device()
        )
        : inference_algorithm<Network_>(bn, n_samples, seed)
        , memory(memory)
        , device(device)
        , context(device)
        , queue(context, device)
        , rand(queue, seed)
        {}

        using prob_v = boost::compute::vector<probability_type>;

    protected:
        compute::device device;
        compute::context context;
        compute::command_queue queue;
        compute::default_random_engine rand;
        unsigned long memory;

        /**
         * calculate the number of iterations needed for a complete simulation without exceeding the boundary set
         * by the user
         * @param bn network
         * @return pair<number of samples per iteration, number of iteration>
         */
        std::pair<unsigned long, unsigned long> calculate_iterations()
        {
            unsigned long sample_p = this->memory / (bn.number_of_variables() * sizeof(probability_type) + 3 * sizeof(uint16_t)) * MEMORY_SLACK / 100;
            if(sample_p < this->nsamples)
                return {sample_p, this->nsamples / sample_p};
            else
                return {this->nsamples, 1};
        }

        std::vector<probability_type> accumulate_cpt(unsigned long v_id, baylib::cow::cpt<probability_type> cpt) {
            auto factory = baylib::condition_factory(bn, v_id, bn.parents_of(v_id));
            std::vector<probability_type> flat_cpt{};
            unsigned int n_states = bn[v_id].table().number_of_states();
            do {
                auto temp = cpt[factory.get()];
                flat_cpt.insert(flat_cpt.end(), temp.begin(), temp.end());
            } while (factory.has_next());

            for (baylib::state_t i = 0; i < flat_cpt.size(); i += n_states)
                for (baylib::state_t j = 1; j < n_states - 1; j++)
                    flat_cpt[i + j] += flat_cpt[i + j - 1];
            return flat_cpt;
        }

        /**
         * Simulations of a specific node using opencl
         * @param cpt cpt of the node
         * @param parents_result results of previous simulate_node calls
         * @param dim number of samples of the simulation
         * @return result of the simulation
         */
        bcvec simulate_node(
                unsigned long v_id,
                const cow::cpt<probability_type> &cpt,
                std::vector<bcvec*> &parents_result,
                int dim
        )
        {
            std::vector<probability_type> flat_cpt_accum = accumulate_cpt(v_id, cpt);
            bcvec result(dim, cpt.number_of_states(), context);
            prob_v device_cpt(flat_cpt_accum.size(), context);
            prob_v threshold_vec(dim, context);
            prob_v random_vec(dim, context);
            compute::uniform_real_distribution<probability_type> distribution(0, 1);
            compute::vector<int> index_vec(dim, context);

            // Async copy of the cpt in gpu memory
            compute::copy(flat_cpt_accum.begin(), flat_cpt_accum.end(), device_cpt.begin(), queue);

            // cycle for deducing the row of the cpt given the parents state in the previous simulation
            if(parents_result.empty())
                compute::fill(index_vec.begin(), index_vec.end(), 0, queue);
            else {
                unsigned int coeff = bn[v_id].table().number_of_states();
                for (int i = 0; i < parents_result.size(); i++) {
                    if (i == 0)
                        compute::transform(parents_result[i]->state.begin(),
                                           parents_result[i]->state.end(),
                                           index_vec.begin(),
                                           _1 * coeff, queue);
                    else
                        compute::transform(parents_result[i]->state.begin(),
                                           parents_result[i]->state.end(),
                                           index_vec.begin(),
                                           index_vec.begin(),
                                           _1 * coeff + _2, queue);
                    coeff *= parents_result[i]->cardinality;
                }
            }

            // get the threshold corresponding to the specific row of the cpt for every single simulation
            compute::gather(index_vec.begin(),
                            index_vec.end(),
                            device_cpt.begin(),
                            threshold_vec.begin(), queue);


            // generate random vector
            distribution.generate(random_vec.begin(),
                                  random_vec.end(),
                                  rand, queue);

            // confront the random vector with the threshold
            compute::transform(random_vec.begin(),
                               random_vec.end(),
                               threshold_vec.begin(),
                               result.state.begin(),
                               _1 > _2,
                               queue);

            // generalization in case of more than 2 states
            for (int i = 0; i + 2 < bn[v_id].table().number_of_states(); i++) {
                compute::vector<int> temp(dim, context);
                compute::transform(index_vec.begin(),
                                   index_vec.end(),
                                   index_vec.begin(),
                                   _1 + 1, queue);
                compute::gather(index_vec.begin(),
                                index_vec.end(),
                                device_cpt.begin(),
                                threshold_vec.begin(), queue);
                compute::transform(random_vec.begin(),
                                   random_vec.end(),
                                   threshold_vec.begin(),
                                   temp.begin(),
                                   _1 > _2, queue);
                compute::transform(temp.begin(),
                                   temp.end(),
                                   result.state.begin(),
                                   result.state.begin(),
                                   _1 + _2, queue);
            }

            return result;
        }
    };
}

#endif //BAYLIB_VECTORIZED_INFERENCE_OPENCL_HPP