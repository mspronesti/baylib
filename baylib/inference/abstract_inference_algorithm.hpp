#ifndef BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP
#define BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP

#define CL_TARGET_OPENCL_VERSION 220

#include <baylib/network/bayesian_utils.hpp>
#include <baylib/probability/marginal_distribution.hpp>
#include <baylib/tools/random/random_generator.hpp>
#include <baylib/tools/gpu/gpu_utils.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <future>
#include <baylib/baylib_concepts.hpp>

//! \file abstract_inference_algorithm.hpp
//! \brief Abstract classes for stocastic algorithms

namespace baylib {
    namespace inference {
        /**
        * This class models a generic approximate
        * inference algorithm for discrete Bayesian
        * networks
        * @tparam Network_ the type of bayesian network
        */
        template <BNetDerived Network_>
        class inference_algorithm {
        public:
            typedef Network_ network_type;
            typedef typename network_type::variable_type variable_type;
            typedef typename variable_type::probability_type probability_type;

            /**
             * The abstract inference algorithm
             * only receives hyperparameters in the
             * constructor
             * @param nsamples : number of samples
             * @param nthreads : number of threads (default: 1)
             * @param seed     : custom seed for the generator (default: 0)
             */
            explicit inference_algorithm(
                    const network_type & bn,
                    unsigned long nsamples,
                    unsigned int seed = 0
            )
            : bn(bn)
            , nsamples(nsamples)
            , seed(seed)
            {}

            virtual ~inference_algorithm() = default;

            /**
             * Main method of the inference algorithm. Receives the bayesian network
             * but doesn't store it anywhere. It only uses its facilities to perform
             * inference thus avoiding unwanted copies and allowing reusability of the
             * same algorithm object
             * @param bn  : bayesian network
             * @return    : the marginal distributions
             */
            virtual baylib::marginal_distribution<probability_type> make_inference () = 0;

            void set_number_of_samples(unsigned long _nsamples) { nsamples = _nsamples; }

            void set_seed(unsigned int _seed) { seed = _seed; }

        protected:
            const network_type & bn;
            unsigned long nsamples;
            unsigned int seed;
        };

        /**
         * This class models an approximate inference algorithm
         * parallelized with C++ threads.
         * Its make_inference employs the well-known approach of splitting
         * the sampling work over the number of threads and merging the results
         * @tparam Network_  : the type of bayesian network
         */
        template < BNetDerived Network_ >
        class parallel_inference_algorithm : public inference_algorithm<Network_>
        {
        public:
            typedef Network_ network_type;
            using typename inference_algorithm<Network_>::probability_type;
            using  inference_algorithm<Network_>::bn;

            explicit parallel_inference_algorithm(
                    const network_type & bn,
                    unsigned long nsamples,
                    unsigned int nthreads = 1,
                    unsigned int seed = 0
            )
            : inference_algorithm<Network_>(bn, nsamples, seed)
            {
                set_number_of_threads(nthreads);
            }

            /**
             * Models the standard approach towards MCMC parallelization,
             * i.e. assigns the sampling step to the number of available threads
             * and eventually merges the results
             * @param bn : bayesian network graph
             * @return   : the marginal distribution of the variables post inference
             */
            baylib::marginal_distribution<probability_type> make_inference() override
            {
                typedef std::future<baylib::marginal_distribution<probability_type>> result;
                BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                          [this](auto &var) { return baylib::cpt_filled_out(bn, var.id()); }),
                              "conditional probability tables must be properly filled to"
                              " run an inference algorithm",
                              std::runtime_error)

                baylib::marginal_distribution<probability_type> inference_result(bn.begin(), bn.end());
                std::vector<result> results;
                baylib::seed_factory sf(nthreads, this->seed);

                auto job = [this](ulong samples_per_thread, uint seed) {
                    return sample_step(samples_per_thread, seed);
                };

                ulong samples_per_thread = this->nsamples / nthreads;
                // assigning jobs
                for (uint i = 0; i < nthreads - 1; ++i)
                    results.emplace_back(std::async(job, samples_per_thread, sf.get_new()));

                // last thread (doing the extra samples if nsamples % nthreads != 0)
                ulong left_samples = this->nsamples - (nthreads - 1) * samples_per_thread;
                results.emplace_back(std::async(job, samples_per_thread, sf.get_new()));

                // accumulate results of each parallel execution
                for (auto &res: results)
                    inference_result += res.get();

                // normalize the distribution before retrieving it
                inference_result.normalize();
                return inference_result;
            }

            void set_number_of_threads(unsigned int _nthreads) {
                nthreads = _nthreads >= std::thread::hardware_concurrency() ?
                           std::thread::hardware_concurrency() : _nthreads > 0 ?
                                                                 _nthreads : 1;
            }

        protected:
            virtual baylib::marginal_distribution<probability_type> sample_step (
                    unsigned long nsamples_per_step,
                    unsigned int seed
            ) = 0;

            unsigned int nthreads;
        };


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
            using typename inference_algorithm<Network_>::network_type;
            using typename inference_algorithm<Network_>::probability_type;
            using  inference_algorithm<Network_>::bn;

            vectorized_inference_algorithm(
                    const network_type & bn,
                    ulong n_samples,
                    size_t memory,
                    uint seed = 0,
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
            size_t memory;

            /**
             * calculate the number of iterations needed for a complete simulation without exceeding the boundary set
             * by the user
             * @param bn network
             * @return pair<number of samples per iteration, number of iteration>
             */
            std::pair<ulong, ulong> calculate_iterations()
            {
                ulong sample_p = this->memory / (bn.number_of_variables() * sizeof(probability_type) + 3 * sizeof(cl_ushort)) * MEMORY_SLACK / 100;
                if(sample_p < this->nsamples)
                    return {sample_p, this->nsamples / sample_p};
                else
                    return {this->nsamples, 1};
            }

            std::vector<probability_type> accumulate_cpt(ulong v_id, baylib::cow::cpt<probability_type> cpt) {
                auto factory = baylib::condition_factory(bn, v_id, bn.parents_of(v_id));
                std::vector<probability_type> flat_cpt{};
                uint n_states = bn[v_id].table().number_of_states();
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
                    ulong v_id,
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
                    uint coeff = bn[v_id].table().number_of_states();
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
    } // namespace inference
} // namespace baylib

#endif //BAYLIB_ABSTRACT_INFERENCE_ALGORITHM_HPP

