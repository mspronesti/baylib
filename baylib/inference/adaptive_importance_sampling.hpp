//
// Created by paolo on 11/09/21.
//

#ifndef BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_HPP
#define BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_HPP

#define L 1000

#define SCORE(var, cond, state) \
icptvec[var][cond][state] / bn[var].table()[cond][state]

#include <inference/abstract_inference_algorithm.hpp>
#include <probability/icpt.hpp>

namespace bn {
    namespace inference{



        template <typename Probability, typename Generator = std::mt19937>
        class adaptive_importance_sampling: public inference_algorithm<Probability> {
            using icpt_vector = std::vector<icpt<Probability>> ;

        public:

            explicit adaptive_importance_sampling(
                    ulong nsamples,
                    uint nthreads = 1,
                    uint seed = 0)
                    : inference_algorithm<Probability>(nsamples, nthreads, seed),
                            w_k(0.0){
            }

            bn::marginal_distribution<Probability> make_inference (
                    const bayesian_network<Probability> &bn
                    ) override{
                icpt_vector icptvec{};
                parent_of_evidence = std::vector(bn.number_of_variables(), false);

                for (int v_id = 0; v_id < bn.number_of_variables(); ++v_id){
                    icptvec.emplace_back(icpt<Probability>(bn[v_id].table()));

                    if (bn[v_id].is_evidence())
                        mark_ancestors(bn, v_id);
                }

                return learn_icpt(bn, icptvec);
            }


        private:

            double w_k;
            std::vector<bool> parent_of_evidence;

            void mark_ancestors(const bn::bayesian_network<Probability> bn, ulong v_id){
                for (ulong p_id: bn.parents_of(v_id)) {
                    if(parent_of_evidence[p_id])
                        continue;
                    parent_of_evidence[p_id] = true;
                    mark_ancestors(bn, p_id);
                }
            }

            marginal_distribution<Probability> learn_icpt(const bn::bayesian_network<Probability> &bn,
                                                          icpt_vector & icptvec){

                bn::random_generator<Probability, Generator> rnd_gen(0);
                const auto vertex_queue = bn::sampling_order(bn);
                bn::marginal_distribution<Probability> result(bn.begin(), bn.end());
                std::vector<std::vector<uint>> graph_state(bn.number_of_variables(), std::vector<uint>(L));


                for (int i = 0; i < this->nsamples; ++i) {
                    if(i % L == 0 && i != 0){
                        learn(graph_state, bn, icptvec);
                        graph_state = std::vector(bn.number_of_variables(), std::vector<uint>(L));
                        w_k += 0.2;
                    }

                    for (ulong v: vertex_queue) {

                        if(bn[v].is_evidence()){
                            graph_state[v][i % L] = bn[v].evidence_state();
                            result[v][bn[v].evidence_state()] += w_k;
                            continue;
                        }

                        const Probability p = rnd_gen();
                        std::vector<Probability> weight;
                        bn::condition parents_state_cond;

                        for(auto par : bn.parents_of(v))
                            parents_state_cond.add(
                                    bn[par].name(),
                                    graph_state[par][ i % L]
                                    );

                        weight = icptvec[v][parents_state_cond];
                        ulong sample = make_random_by_weight(p, weight);
                        graph_state[v][i % L] = sample;


                        if(w_k != 0.0 && !bn[v].is_evidence()){
                            result[v][sample] += w_k;
                        }

                    }
                }
                result.normalize();
                return result;

            }



            void learn(const std::vector<std::vector<uint>>& graph_state, const bn::bayesian_network<Probability>& bn,
                       icpt_vector& icptvec){
                float learning_rate = 0.5;
                Probability evidence_score;
                std::vector<Probability> valid_sample(L, 1.);
                std::vector<Probability> prob_sample(L, 1.);

                for(int i=0; i<L; ++i){
                    bool flag=true;
                    for(int v_id=0; v_id<bn.number_of_variables(); ++v_id){

                        condition cond;
                        for(auto p_id : bn.parents_of(v_id))
                            cond.add(bn[p_id].name(), graph_state[p_id][i]);
                        if(bn[v_id].is_evidence()){
                            valid_sample[i] *= bn[v_id].table()[cond][bn[v_id].evidence_state()];
                        }else{
                            valid_sample[i] *= bn[v_id].table()[cond][graph_state[v_id][i]]/icptvec[v_id][cond][graph_state[v_id][i]];
                        }
                    }
                }

                for (int v_id=0; v_id < graph_state.size(); ++v_id) {
                    if(parent_of_evidence[v_id] && !bn[v_id].is_evidence()){
                        icpt<Probability> temp_icpt;
                        temp_icpt = icpt(bn[v_id].table(), true);
                        for (int i = 0; i < L; ++i) {
                                condition cond;
                                auto sample = graph_state[v_id][i];
                                for(auto p_id : bn.parents_of(v_id))
                                    cond.add(bn[p_id].name(), graph_state[p_id][i]);
                                temp_icpt[cond][sample] += valid_sample[i];
                        }
                        temp_icpt.normalize();
                        icptvec[v_id].absorb(temp_icpt, learning_rate);
                    }
                }
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

#endif //BAYLIB_ADAPTIVE_IMPORTANCE_SAMPLING_HPP
