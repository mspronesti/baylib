//
// Created by elle on 18/08/21.
//

#ifndef BAYLIB_GIBBS_SAMPLING_HPP
#define BAYLIB_GIBBS_SAMPLING_HPP

#include <baylib/probability/marginal_distribution.hpp>
#include <baylib/inference/abstract_inference_algorithm.h>

#include <algorithm>
#include <random>
#include <shared_mutex>
#include <future>
#include <tools/generator_wrapper.hpp>

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
                const bn::bayesian_network<Probability> &bn,
                ulong nvars,
                ulong nsamples,
                uint seed=0
            )
            : nsamples(nsamples)
            , bn(bn)
            , nvars(nvars)
            , var_state_values(std::vector<bn::state_t>(nvars, 0))
            , gen(seed)
            {}

            std::vector<std::pair<ulong,ulong>> sample(){
                for(ulong i = 0; i < nsamples; ++i)
                    for(ulong n = 0; n < nvars; ++n)
                    {
                        auto& var = bn::thread_safe::get_variable(bn, n, m); //thread_safe_get_var(n);
                        auto res = sample_single_variable(var);
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

            const bn::bayesian_network<Probability> &bn;
            generator_wrapper<Probability, Generator> gen;
            ulong nsamples;
            ulong nvars;
            std::shared_mutex m;

            /** --- private members --- **/
            std::pair<ulong, ulong> sample_single_variable( bn::random_variable<Probability> & var )
            {
                ulong index;
                ulong states_size;
                {
                    std::scoped_lock sl{m};
                    index = var.id();
                    states_size = var.states().size();
                }

                auto samples = std::vector<Probability>(states_size, 0.0);
                for(ulong i = 0; i < samples.size(); ++i){
                    var_state_values[index] = i;

                    samples[i] = get_probability(index);
                }

                // normalize
                Probability sum = std::accumulate(samples.begin(), samples.end(), 0.0);
                std::for_each(samples.begin(), samples.end(), [sum](auto & val){
                    val /= sum;
                });

                Probability prob = gen();
                ulong j;
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
                // builds a condition using parents and
                // their states
                for(auto & p : bn.parents_of(n))
                    c.add(
                            bn[p].name(),
                            var_state_values[p]
                            );

                const auto& cpt = bn[n].table();
                return  cpt[c][var_state_values[n]];
            }
        };


        /** ===== Gibbs sampling Algorithm ===
         *
         * This class represents the Gibbs Sampling approximate
         * inference algorithm for discrete Bayesian Networks.
         * It's based on the Gibbs sampler.
         * @tparam Probability : the type expressing the probability
         * @tparam Generator  : the random generator
         *                     (default Mersenne Twister pseudo-random generator)
         */
        template <typename Probability, typename Generator=std::mt19937>
        class gibbs_sampling : abstract_inference_algorithm<Probability>{
        public:
            explicit gibbs_sampling(ulong nsamples, uint nthreads, uint seed = 0)
            : abstract_inference_algorithm<Probability>(nsamples, nthreads, seed), counter(0)
            {};

            bn::marginal_distribution<Probability> make_inference(
                   const bayesian_network<Probability> &bn
            ) override
            {
                marginal_distribution<Probability> marginal_distr(bn.variables());
                auto vars  = bn.variables();
                BAYLIB_ASSERT(std::all_of(vars.begin(), vars.end(),
                                          [](auto &var){ return bn::cpt_filled_out(var); }),
                              "conditional probability tables must be properly filled to"
                              " run logic_sampling inference algorithm",
                              std::runtime_error)
                ulong nvars = bn.number_of_variables();
                ulong samples_per_thread = this->nsamples / this->nthreads;

                for(auto i = 0; i < this->nthreads - 1; ++i){
                    assign_worker(bn, nvars, samples_per_thread);
                }

                // last thread (if nsamples % nthread != 0, last thread is gonna do the extra samples)
                assign_worker(bn, nvars, this->nsamples - (this->nthreads - 1) * samples_per_thread);


                for(auto & res : results)
                {
                    for(auto  [var_id, state_val] : res.get())
                        ++marginal_distr[var_id][state_val];
                }
                marginal_distr /= (Probability)this->nsamples;
                return marginal_distr;
            }


        private:
            void assign_worker(
                const bayesian_network<Probability> &bn,
                ulong nvars,
                ulong samples_per_thread
            )
            {
                auto job = [this, &bn](ulong nvars, ulong samples_per_thread){
                    auto worker = gibbs_worker<Probability, Generator>(bn, nvars, samples_per_thread, (this->seed + counter++) * 2654435761 % 2^32);
                    return worker.sample();
                };

                results.push_back(std::async(job, nvars, samples_per_thread));
            }

            std::shared_mutex m;
            std::vector<result> results;
            uint counter;
        };
    } } // namespace bn

#endif //BAYLIB_GIBBS_SAMPLING_HPP
