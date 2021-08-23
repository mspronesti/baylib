//
// Created by elle on 18/08/21.
//

#ifndef BAYLIB_GIBBS_SAMPLING_HPP
#define BAYLIB_GIBBS_SAMPLING_HPP

#include <baylib/network/bayesian_network.hpp>
#include <baylib/probability/marginal_distribution.hpp>

#include <algorithm>
#include <random>
#include <shared_mutex>
#include <future>

namespace bn {
    namespace inference {
        typedef std::vector<std::pair<ulong, ulong>> vec_pair;
        typedef std::shared_future<vec_pair> result;

        template <typename Probability, typename Generator=std::mt19937>
        class gibbs_worker{
            /**
             * This class represents a single worker
             * i.e. a thread performing a complete simulation
             * of the bayesian network graph nsamples time.
             * Each Gibbs sampler is responsible of tot_nsamples/n_workers
             * simulations
             */
        public:
            gibbs_worker(
                bn::bayesian_network<Probability> &bn,
                ulong nvars,
                ulong nsamples
            )
            : nsamples(nsamples)
            , bn(bn)
            , nvars(nvars)
            {
                // random initial state values
                for(ulong i = 0; i < nvars; ++i)
                    var_state_values.push_back(
                            rand_from_distribution(0, bn[i].states().size())
                    );
            }

            std::vector<std::pair<ulong,ulong>> sample(){
                for(ulong i = 0; i < nsamples; ++i)
                    for(ulong n = 0; n < nvars; ++n)
                    {
                        auto& var = thread_safe_get_var(n);
                        auto children = thread_safe_var_children(n);
                        auto res = sample_single_variable(var, children);
                        marginal_pairs.push_back(res);
                    }

                return marginal_pairs;
            }

        private:
            // each pair contains the (vid, state_val) couple
            // to be incremented in the marginal distribution
            std::vector<std::pair<ulong,ulong>> marginal_pairs;

            // contains, for each variable, the current state value
            std::vector<bn::state_t> var_state_values;

            bn::bayesian_network<Probability> &bn;
            ulong nsamples;
            ulong nvars;
            std::shared_mutex m;

            /** --- private members --- **/
            std::pair<ulong, ulong> sample_single_variable(
                    bn::random_variable<Probability> & var,
                    const std::vector<ulong> &children
            )
            {
                ulong index;
                ulong states_size;
                {
                    std::scoped_lock sl{m};
                    index = var.id();
                    states_size = var.states().size();
                }

                auto samples = std::vector<Probability>(states_size, 0.0);
                for(long i = 0; i < samples.size(); ++i){
                    var_state_values[index] = i;

                    samples[i] = get_probability(index);
                    for (auto child : children)
                        samples[i] *= get_probability(child);
                }

                // normalize
                Probability sum = std::accumulate(samples.begin(), samples.end(), 0.0);
                std::for_each(samples.begin(), samples.end(), [sum](auto & val){
                    val /= sum;
                });

                Probability prob = rand_from_distribution();
                long j;
                for (j = 0; j < samples.size() - 1; ++j) {
                    if (prob <= samples[j])
                        break;
                    else
                        prob -= samples[j];
                }

                var_state_values[index] = j;
                return {index, j};
            }

            Probability get_probability ( const unsigned long n )
            {
                std::scoped_lock sl{m};
                bn::condition c;
                for(auto & p : bn.parents_of(n))
                    c.add(
                            bn[p].name(),
                            var_state_values[p]
                            );

                auto& cpt = bn[n].table();
                return  cpt[c][var_state_values[n]];
            }

            Probability rand_from_distribution(Probability from = 0.0, Probability to = 1.0)
            {
                thread_local static Generator gen(std::random_device{}());

                using dist_type = typename std::conditional
                        <
                        std::is_integral<Probability>::value
                        , std::uniform_int_distribution<Probability>
                        , std::uniform_real_distribution<Probability>
                        >::type;

                thread_local static dist_type dist;
                return dist(gen, typename dist_type::param_type{from, to});
            }

            bn::random_variable<Probability> & thread_safe_get_var(ulong vid){
                std::scoped_lock sl{m};
                return bn[vid];
            }

            std::vector<ulong> thread_safe_var_children(ulong vid){
                std::scoped_lock sl{m};
                return bn.children_of(vid);
            }
        };


        /** ===== Gibbs sampling Algorithm ===
         *
         * This class represents the Gibbs Sampling approximate
         * inference algorithm for discrete Bayesian Networks.
         * It's based on the Gibbs sampler.
         * @tparam Probability : the type expressing the probability
         * @tparam Generator : the random generator
         *                     (default Mersenne Twister pseudo-random generator)
         */
        template <typename Probability, typename Generator=std::mt19937>
        class gibbs_sampling {
        public:
            explicit gibbs_sampling(const bn::bayesian_network<Probability> &bn)
            : bn(bn)
            , marginal_distr(bn.variables())
            {
                auto vars  = bn.variables();
                BAYLIB_ASSERT(std::all_of(vars.begin(), vars.end(),
                                          [](auto &var){ return bn::cpt_filled_out(var); }),
                              "conditional probability tables must be properly filled to"
                              " run logic_sampling inference algorithm",
                              std::runtime_error)
            };

            bn::marginal_distribution<Probability> inferenciate(
                    unsigned long nsamples,
                    unsigned int nthreads = 1
            )
            {
                ulong nvars = bn.number_of_variables();
                ulong samples_per_thread = nsamples / nthreads;

                for(auto i = 0; i < nthreads - 1; ++i){
                    assign_worker(nvars, samples_per_thread);
                }

                // last thread (if nsamples % nthread != 0, last thread is gonna do the extra samples)
                assign_worker(nvars, nsamples - (nthreads - 1) * samples_per_thread);


                for(auto & res : results)
                {
                    for(auto  [var_id, state_val] : res.get())
                        ++marginal_distr[var_id][state_val];
                }
                marginal_distr /= (Probability)nsamples;
                return marginal_distr;
            }

            bn::marginal_distribution<Probability> inference_result() const {
                return marginal_distr;
            }

        private:
            void assign_worker(
                ulong nvars,
                ulong samples_per_thread
            )
            {
                auto job = [this](ulong nvars, ulong samples_per_thread){
                    auto worker = gibbs_worker<Probability, Generator>(bn, nvars, samples_per_thread);
                    return worker.sample();
                };

                results.push_back(std::async(job, nvars, samples_per_thread));
            }

            std::shared_mutex m;
            bn::bayesian_network<Probability> bn;
            bn::marginal_distribution<Probability> marginal_distr;
            std::vector<result> results;
        };
    } // namespace inference
} // namespace bn

#endif //BAYLIB_GIBBS_SAMPLING_HPP
