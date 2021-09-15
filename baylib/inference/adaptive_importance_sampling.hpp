//
// Created by paolo on 11/09/21.
//

#ifndef BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_HPP
#define BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_HPP

#define CL_TARGET_OPENCL_VERSION 220

#include <inference/abstract_inference_algorithm.hpp>
#include <probability/icpt.hpp>
#include <probability/cpt.hpp>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <utility>
#include <execution>
#include <tools/gpu/gpu_utils.hpp>
#include <network/bayesian_utils.hpp>

namespace bn {
    namespace inference{

        template <typename Probability, typename Generator = std::mt19937>
        class adaptive_importance_sampling: public vectorized_inference_algorithm<Probability> {
        using icpt_vector = std::vector<icpt<Probability>>;
        using simulation_matrix = std::vector<std::vector<uint>>;

        public:

            /**
             * Adaptive Sampling algorithm constructor
             * @param nsamples : number of samples
             * @param memory : maximum amount of memory usable by the opencl device
             * @param initial_learning_rate : starting learning rate, must be higher then the final one
             * @param final_learning_rate : final learning rate, must be lower then the initial one
             * @param learning_step : number of iterations after witch the learning phase is stopped
             * @param seed : seed for the random generators
             * @param device : opencl device used for the simulation
             */
            explicit adaptive_importance_sampling(
                    ulong nsamples,
                    size_t memory,
                    double initial_learning_rate = 1,
                    double final_learning_rate = 0.05,
                    uint learning_step = 1000,
                    uint seed = 0,
                    const compute::device& device = compute::system::default_device()):
                        vectorized_inference_algorithm<Probability>(nsamples, memory, seed, device),
                        w_k(1),
                        initial_learning_rate(initial_learning_rate),
                        final_learning_rate(final_learning_rate),
                        learning_cutoff(0.01),
                        learning_step(learning_step){
            }

            bn::marginal_distribution<Probability> make_inference (
                    const bayesian_network<Probability> &bn
                    ) override{
                icpt_vector icptvec{};
                auto result = marginal_distribution<Probability>(bn.begin(), bn.end());
                bool flag = false;
                for (int v_id = 0; v_id < bn.number_of_variables(); ++v_id){
                    icptvec.emplace_back(icpt<Probability>(bn[v_id].table()));
                    if(bn[v_id].is_evidence()){
                        result[v_id][bn[v_id].evidence_state()] = 1;
                        flag = true;
                    }
                }
                if(flag){
                    ancestors = ancestors_of_evidence(bn);
                    result += learn_icpt(bn, icptvec);
                }
                result +=  gpu_simulation(icptvec, bn);
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


            marginal_distribution<Probability> learn_icpt(const bn::bayesian_network<Probability> &bn,
                                                          icpt_vector & icptvec){

                bn::random_generator<Probability, Generator> rnd_gen(0);
                const auto vertex_queue = bn::sampling_order(bn);
                bn::marginal_distribution<Probability> result(bn.begin(), bn.end());
                simulation_matrix graph_state(bn.number_of_variables(), std::vector<uint>(learning_step));
                double k = 0;
                double max_k = this->nsamples/learning_step;

                for (int i = 0; i < this->nsamples; ++i) {
                    if(i % learning_step == 0 && i != 0){

                        double learning_rate = initial_learning_rate *std::pow(final_learning_rate / initial_learning_rate,
                                                                               k / max_k);
                        double difference = absorb_samples(graph_state, bn, icptvec, learning_rate);
                        if(difference < learning_cutoff)
                            break;
                        graph_state = std::vector(bn.number_of_variables(), std::vector<uint>(learning_step));
                        w_k++;
                        k++;
                    }

                    for (ulong v: ancestors) {

                        if(bn[v].is_evidence()){
                            graph_state[v][i % learning_step] = bn[v].evidence_state();
                            continue;
                        }

                        const Probability p = rnd_gen();
                        std::vector<Probability> weight;
                        bn::condition parents_state_cond;

                        for(auto par : bn.parents_of(v))
                            parents_state_cond.add(
                                    bn[par].name(),
                                    graph_state[par][i % learning_step]
                                    );

                        weight = icptvec[v][parents_state_cond];
                        ulong sample = make_random_by_weight(p, weight);
                        graph_state[v][i % learning_step] = sample;

                        result[v][sample] += w_k;

                    }
                }
                result.normalize();
                return result;

            }



            double absorb_samples(const simulation_matrix & graph_state,
                                  const bn::bayesian_network<Probability> & bn,
                                  icpt_vector & icptvec,
                                  double learning_rate){
                Probability evidence_score;
                std::vector<Probability> sample_weight(learning_step, 1.);

                double max_distance = 0.;

                for(int i=0; i<learning_step; ++i){
                    for(ulong v_id: ancestors){
                        condition cond;
                        auto& icpt = icptvec[v_id];
                        auto& cpt = bn[v_id].table();
                        auto& sample_state = graph_state[v_id][i];

                        for(auto p_id : bn.parents_of(v_id))
                            cond.add(bn[p_id].name(), graph_state[p_id][i]);
                        if(bn[v_id].is_evidence()){
                            sample_weight[i] *= cpt[cond][bn[v_id].evidence_state()];
                        }else{
                            sample_weight[i] *= cpt[cond][sample_state] / icpt[cond][sample_state];
                        }

                    }
                }

                for(ulong v_id: ancestors){
                    icpt<Probability> temp_icpt;
                    auto& original_cpt = icptvec[v_id];
                    temp_icpt = icpt(bn[v_id].table(), true);
                    for (int i = 0; i < learning_step; ++i) {
                            condition cond;
                            auto sample = graph_state[v_id][i];
                            for(auto p_id : bn.parents_of(v_id))
                                cond.add(bn[p_id].name(), graph_state[p_id][i]);
                            temp_icpt[cond][sample] += sample_weight[i];
                    }
                    temp_icpt.normalize();
                    double distance = original_cpt.absorb(temp_icpt, learning_rate);
                    max_distance = distance > max_distance ? distance : max_distance;
                }
                return max_distance;
            }

            marginal_distribution<Probability> gpu_simulation(icpt_vector icpt_vec, const bayesian_network<Probability>& bn){
                int niter = 1;
                marginal_distribution<Probability> marginal_result(bn.begin(), bn.end());
                std::vector<bcvec> result_container(bn.number_of_variables());
                marginal_distribution<Probability> temp(bn.begin(), bn.end());

                for(ulong v : sampling_order(bn)) {
                    if(bn[v].is_evidence()){
                        auto res = compute::vector<cl_short>(this->nsamples, (cl_short)bn[v].evidence_state() ,this->queue);
                        result_container[v] = bcvec(res,
                                bn[v].states().size(),
                                bn.children_of(v).size());
                    }
                    else{
                        std::vector<bcvec> parents_result;
                        auto parents = bn[v].parents_info.names();
                        std::reverse(parents.begin(), parents.end());
                        for (auto p : parents) {
                            parents_result.push_back(result_container[bn.index_of(p)]);
                        }

                        auto result = this->simulate_node(icpt_vec[v] , parents_result, this->nsamples);
                        result_container[v] = bcvec(result, bn[v].states().size(), bn.children_of(v).size());

                        auto accumulated_result = compute_result_general(result_container[v]);

                        for(int ix=0; ix< accumulated_result.size(); ix++)
                            marginal_result[v][ix] += accumulated_result[ix];
                    }
                    for(auto p: bn.parents_of(v)){
                        result_container[p].set_use();
                    }
                }
                return marginal_result;
            }

            std::vector<ulong> compute_result_general(bcvec& res)
            {
                std::vector<ulong> acc_res(res.cardinality);
                for (bn::state_t i = 0; i < res.cardinality; ++i) {
                    acc_res[i] = w_k * compute::count(res.get_states().begin(), res.get_states().end(), i, this->queue);
                }
                return acc_res;
            }


            uint make_random_by_weight(
                    const Probability p,
                    const std::vector<Probability> & weight
                    ){

                Probability total = 0.0;
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

#endif //BAYIB_ADAPTIVE_IMPORTANCE_SAMPING_HPP
