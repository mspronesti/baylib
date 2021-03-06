//
// Created by paolo on 11/09/21.
//

#ifndef BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_OPENCL_HPP
#define BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_OPENCL_HPP

#define CL_TARGET_OPENCL_VERSION 220

#include <baylib/inference/opencl/vectorized_inference_opencl.hpp>
#include <baylib/probability/icpt.hpp>
#include <baylib/probability/cpt.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <utility>
#include <execution>
#include <baylib/tools/gpu/gpu_utils.hpp>
#include <baylib/network/bayesian_utils.hpp>

//! \file adaptive_importance_sampling_opencl.hpp
//! \brief Adaptive sampling implementation with opencl optimization and multi-thread support

namespace baylib {
    namespace inference{

        /**
         * This class represents a possible implementation of the Adaptive Importance Sampling algorithm.
         * The implementation exploits two different types of parallelization, multithreading with the tbb
         * library in the first approximation part and GPGPU with boost::compute in the sampling part.
         * The steps of the approximation part are:
         * 1. create the sub-graph of the nodes that are ancestors of evidence nodes
         * 2. sort the nodes in topological order
         * 3. simulate the sub-graph
         * 4. approximate a new cpt for each node given the simulations
         * 5. repeat from 3 until cpts don't change anymore or a maximum number of samples is reached
         * The steps of the sampling part are:
         * 1. sort all nodes in topological order
         * 2. simulate all root nodes
         * 3. simulate all other nodes given their parents
         * 4. each evidence node simulation is set to the evidence value
         * 5. estimate the marginal probabilities from the simulations
         * @tparam Network_   : the type of bayesian network (must inherit from baylib::bayesian_net)
         * @tparam Generator_ : the type of random generator
         *                  (default Mersenne Twister pseudo-random generator)
         */
        template <
                BNetDerived Network_,
                typename Generator_ = std::mt19937
                >
        class adaptive_importance_sampling_opencl: public baylib::inference::vectorized_inference_algorithm<Network_>
        {
            using typename vectorized_inference_algorithm<Network_>::probability_type;
            using vectorized_inference_algorithm<Network_>::bn;
            typedef std::vector<cow::icpt<probability_type>> icpt_vector;
            typedef std::vector<std::vector<uint>>  simulation_matrix;
            typedef Network_ network_type;

        public:

            /**
             * Adaptive Sampling algorithm constructor
             * @param nsamples : number of samples
             * @param memory : maximum amount of memory usable by the opencl device
             * @param n_threads : maximum number of threads in the initial learning phasae
             * @param initial_learning_rate : starting learning rate, must be higher then the final one
             * @param final_learning_rate : final learning rate, must be lower then the initial one
             * @param learning_step : number of iterations after witch the learning phase is stopped
             * @param seed : seed for the random generators
             * @param device : opencl device used for the simulation
             */
            explicit adaptive_importance_sampling_opencl(
                const network_type & bn,
                ulong nsamples,
                size_t memory,
                double initial_learning_rate = 1,
                double final_learning_rate = 0.05,
                uint learning_step = 1000,
                uint seed = 0,
                const compute::device& device = compute::system::default_device()
            )
            : vectorized_inference_algorithm<Network_>(bn, nsamples, memory, seed, device)
            , w_k(1)
            , initial_learning_rate(initial_learning_rate)
            , final_learning_rate(final_learning_rate)
            , learning_cutoff(0.005)
            , learning_step(learning_step){}

            /**
             * Inference method for adaptive_sampling,
             * the inference process is divided in 2 steps, first we estimate \f{eqnarray*}P(X|Par(X), E)\f and after
             * that we simulate the whole network and collect the results
             * @param bn network
             * @return marginal distribution
             */
            baylib::marginal_distribution<probability_type> make_inference ()
            {
                BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                          [this](auto &var){ return baylib::cpt_filled_out(bn, var.id()); }),
                              "conditional probability tables must be properly filled to"
                              " run logic_sampling_opencl inference algorithm",
                              std::runtime_error);

                icpt_vector icptvec{};
                auto result = marginal_distribution<probability_type>(bn.begin(), bn.end());
                bool evidence_found = false;
                for (int v_id = 0; v_id < bn.number_of_variables(); ++v_id){
                    icptvec.emplace_back(cow::icpt<probability_type>(bn[v_id].table()));
                    if(bn[v_id].is_evidence()){
                        result[v_id][bn[v_id].evidence_state()] = 1;
                        evidence_found = true;
                    }
                }
                // If no evidence is present the algorithm degenerates to simple
                // logic_sampling_opencl, and we can skip the learning phase
                if(evidence_found){
                    ancestors = ancestors_of_evidence(bn);
                    learn_icpt(icptvec);
                }
                result += gpu_simulation(icptvec, bn);
                result.normalize();
                return result;
            }


        private:
            double w_k;
            std::vector<ulong> ancestors;
            double initial_learning_rate;
            double final_learning_rate;
            double learning_cutoff;
            uint learning_step;


            /**
             * cpts model the distribution \f{eqnarray*}P(X_i|Par(X_i))\f while
             * icpts model the distribution \f{eqnarray*}P(X_i|Par(X_i), E)\f
             * using stochastic simulations we can approximate the icpts starting from the cpts.
             * every #learning_step simulations we stop the sampling and we make an estimation
             * for the icpts, if the new estimation was close to the one we already had we
             * can stop early the simulation.
             * icpts != cpt only if the node is an ancestor of an evidence node, this means that we can
             * avoid simulating all the nodes not in the ancestor set
             * @param bn the network
             * @param icptvec the icpts that will be used and returned by reference
             * @return partial results made from the simulations of the icpts
             */

            void learn_icpt( icpt_vector & icptvec )
            {
                ulong nvars = bn.number_of_variables();
                simulation_matrix graph_state(learning_step);
                std::vector<probability_type> random_vec(learning_step * nvars);
                double k = 0;
                uint max_k = this->nsamples / 2 / learning_step;
                seed_factory factory(1, this->seed);
                baylib::random_generator<probability_type, Generator_> rnd_gen(this->seed);

                for (int i = 0; i < max_k; ++i) {
                    std::future<void> el = std::async([&](){std::generate(
                                                random_vec.begin(),
                                                random_vec.end(),
                                                rnd_gen);});
                    if(i != 0){
                        // Formula for the learning_rate proposed in the paper
                        double learning_rate = initial_learning_rate *
                                std::pow(final_learning_rate / initial_learning_rate,k / (double)max_k);
                        double difference = absorb_samples(graph_state, bn, icptvec, learning_rate);
                        // if the maximum difference was low enough we stop the learning process
                        if(difference < learning_cutoff){
                            break;
                        }
                        k++;
                    }
                    el.get();

                    std::transform( std::execution::par,
                                    boost::counting_iterator<ulong>(0),
                                    boost::counting_iterator<ulong>(graph_state.size()),
                                    graph_state.begin(),
                                    [&](ulong ix)
                                    {
                                    std::vector<uint> local_result(nvars);
                                    ix *= nvars;
                                    for (ulong v: ancestors) {
                                        if (bn[v].is_evidence()) {
                                            local_result[v] = bn[v].evidence_state();
                                            continue;
                                        }
                                        const probability_type p = random_vec[ix];
                                        ix++;
                                        std::vector<probability_type> weight;
                                        baylib::condition parents_state_cond;
                                        for (auto par : bn.parents_of(v))
                                            parents_state_cond.add(
                                                    par,
                                                    local_result[par]
                                                    );
                                        weight = icptvec[v][parents_state_cond];
                                        ulong sample = make_random_by_weight(p, weight);
                                        local_result[v] = sample;
                                    }
                                    return local_result;
                                    });
                }
                this->seed = factory.get_new(); // new seed for gpu simulation
            }


            /**
             * After simulating enough samples we can estimate \f{eqnarray*}P(X_i|Par(X_i),E)\f
             * @param graph_state simulations in matrix format
             * @param bn network
             * @param icptvec icpts to update
             * @param learning_rate learning rate used for updating icpts
             * @return maximum distance that was recorded between pairs of old icpts and new icpts
             */
            double absorb_samples(
                const simulation_matrix & graph_state,
                const network_type & bn,
                icpt_vector & icptvec,
                double learning_rate
            )
            {
                probability_type evidence_score;
                std::vector<probability_type> sample_weight(graph_state.size());

                double max_distance = 0.;

                // Calculate the likelihood of extracting a particular set of samples
                std::transform(std::execution::par_unseq,
                               boost::counting_iterator<uint>(0),
                               boost::counting_iterator<uint>(graph_state.size()),
                               sample_weight.begin(),
                               [&](uint ix)
                               {
                                probability_type weight = 1;
                                for(ulong v_id: ancestors){
                                    condition cond;
                                    auto& icpt = icptvec[v_id];
                                    auto& cpt = bn[v_id].table();
                                    auto& sample_state = graph_state[ix][v_id];

                                    for(auto p_id : bn.parents_of(v_id))
                                        cond.add(p_id, graph_state[ix][p_id]);
                                    if(bn[v_id].is_evidence()){
                                        weight *= cpt[cond][bn[v_id].evidence_state()];
                                    }else{
                                        weight *= cpt[cond][sample_state] / icpt[cond][sample_state];
                                    }
                                }
                                return  weight;});

                // Update the icpts and return the maximum distance
                max_distance =  std::transform_reduce(
                                std::execution::par_unseq,
                                ancestors.begin(),
                                ancestors.end(),
                                0.,
                                [](auto e1, auto e2){return e1 > e2 ? e1 : e2;},
                                [&](auto v_id)
                                {
                                    if(bn[v_id].is_evidence())
                                        return 0.;
                                    auto& original_cpt = icptvec[v_id];
                                    cow::icpt<probability_type> temp_icpt(bn[v_id].table(), true);
                                    for (int i = 0; i < graph_state.size(); ++i) {
                                        condition cond;
                                        auto sample = graph_state[i][v_id];
                                        for(auto p_id : bn.parents_of(v_id))
                                            cond.add(p_id, graph_state[i][p_id]);
                                        temp_icpt[cond][sample] += sample_weight[i];
                                    }
                                    temp_icpt.normalize();
                                    double distance = original_cpt.absorb(temp_icpt, learning_rate);
                                    return distance;
                                });

                return max_distance;
            }

            /**
             * gpu simulation of the network, if we correctly updated \f{eqnarray*}P(X|Par(X))\f to \f{eqnarray*}P(X|Par(X),E)\f no more
             * adjustments to scoring should be necessary and we can simulate easily the rest of the network
             * @param icpt_vec vector of icpts
             * @param bn network
             * @return compressed results of the simulation
             */
            marginal_distribution<probability_type> gpu_simulation(
                    const icpt_vector& icpt_vec,
                    const network_type & bn
            )
            {
                int niter = 1;
                marginal_distribution<probability_type> marginal_result(bn.begin(), bn.end());
                std::vector<bcvec> result_container(bn.number_of_variables());
                marginal_distribution<probability_type> temp(bn.begin(), bn.end());
                auto [gpu_samples, gpu_iter] = this->calculate_iterations();

                for(int i = 0; i < gpu_iter; ++i){
                    for(ulong v : sampling_order(bn)) {
                        if(bn[v].is_evidence()){
                            result_container[v] = bcvec(this->nsamples, bn[v].number_of_states(), this->context);
                            compute::fill(result_container[v].state.begin(), result_container[v].state.end(), bn[v].evidence_state(), this->queue);
                        }
                        else{
                            std::vector<bcvec*> parents_result;
                            auto parents = bn.parents_of(v);

                            for (auto p : parents) {
                                parents_result.push_back(&result_container[p]);
                            }

                            result_container[v] = this->simulate_node(v, icpt_vec[v], parents_result, gpu_samples);

                            auto accumulated_result = compute_result_general(result_container[v]);

                            for(int ix=0; ix< accumulated_result.size(); ix++)
                                marginal_result[v][ix] += accumulated_result[ix];
                        }
                    }
                }
                return marginal_result;
            }

            /**
             * calculate weighted sum of a specific simulation
             * @param res result of the simulation
             * @return accumulated result
             */
            std::vector<ulong> compute_result_general(bcvec& res)
            {
                std::vector<ulong> acc_res(res.cardinality);
                for (baylib::state_t i = 0; i < res.cardinality; ++i) {
                    acc_res[i] = w_k * compute::count(res.state.begin(), res.state.end(), i, this->queue);
                }
                return acc_res;
            }


            /**
             * Given a random probability and the distribution return the sample realization
             * @param p probability_type
             * @param weight discrete distribution
             * @return realization
             */
            uint make_random_by_weight(
                const probability_type p,
                const std::vector<probability_type> & weight
            )
            {
                probability_type total = 0.0;
                for(uint i = 0; i < weight.size(); ++i)
                {
                    auto const old_total = total;
                    total += weight[i];
                    if(old_total <= p && p < total)
                    {
                        return i;
                    }
                }
                return weight.size() - 1;
            }
        };
    }
}

#endif //BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_OPENCL_HPP
