//
// Created by elle on 22/07/21.
//

#ifndef BAYLIB_LOGIC_SAMPLING_HPP
#define BAYLIB_LOGIC_SAMPLING_HPP

#define CL_TARGET_OPENCL_VERSION 220
#define MEMORY_SLACK 0.8

#include <vector>
#include <future>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

#include <baylib/inference/abstract_inference_algorithm.hpp>

namespace bn {
    namespace inference{
        namespace compute = boost::compute;
        using boost::compute::lambda::_1;
        using boost::compute::lambda::_2;

        struct bcvec {
            compute::vector<int> vec;
            ushort cardinality;

            bcvec(
                int dim,
                const compute::context& context,
                compute::command_queue &queue,
                ushort cardinality
            )
            : cardinality(cardinality)
            {
                vec = compute::vector<int>(dim, context);
            }
        };

        /**
         * This class models simulate_node results in both reduced and bvec form.
         * Used by logic_sampling to access bvec of already computed nodes
         * Keeps count of how many times a bvec has been used, and deletes the reference after
         * a bvec has been used out_degree times.
         * Can be added to compressed_result to cast it directly
         * @tparam Probability : the type modeling probability
         **/
        template <typename Probability>
        class uncompressed_result {
        public:
            explicit uncompressed_result(bn::vertex<Probability> nnodes):
            data(std::vector<std::optional<bcvec>>(nnodes)),
            use_count(std::vector<ulong>(nnodes, 0))
            {}

            void put(const std::optional<bcvec>& entry, bn::vertex<Probability> edge, ulong uses) {
                data[edge] = entry;
                use_count[edge] = uses;
            }

            bcvec get(const bn::vertex<Probability> v) {
                auto res = data[v];
                BAYLIB_ASSERT(use_count[v] > 0,
                              "Function get has been called on node "
                              << v << " too many times",
                              std::runtime_error)

                use_count[v]--;
                if (use_count[v] == 0)
                    data[v].reset();

                return res.value();
            }

        private:
            std::vector<std::optional<bcvec>> data;
            std::vector<ulong> use_count;

            std::pair<std::vector<ulong>,bcvec> & operator[] (const bn::vertex<Probability> v)
            {
                return data[v];
            }
        };


        /** ===== Logic Sampling Algorithm ===
        *
        * This class represents the Logic Sampling approximate
        * inference algorithm for discrete Bayesian Networks.
        * The implementation uses boost::compute to exploit GPGPU optimization.
        * All samples are simulated in parallel, for this reason the maximum memory usage
        * tolerable must be specified to avoid filling up the memory of the device in case of large
        * number of samples.
        * @tparam Probability : the type expressing the probability
        **/
        template <typename Probability>
        class logic_sampling : public inference_algorithm<Probability>{
            using prob_v = boost::compute::vector<Probability>;
        public:

            logic_sampling(
                 size_t memory,
                 ulong samples,
                 uint seed = 0,
                 const compute::device &device = compute::system::default_device()
            )
            : inference_algorithm<Probability>(samples)
            , seed(seed)
            , context(compute::context(device))
            , queue(compute::command_queue(context, device))
            , memory(memory)
            { }

            std::pair<std::vector<ulong>, bcvec> simulate_node(
                    const bn::random_variable<Probability> &var,
                    const std::vector<bcvec>& parents_result,
                    int dim,
                    compute::default_random_engine rand_eng
            );

            bn::marginal_distribution<Probability> make_inference(
                    const bn::bayesian_network<Probability> &bn
            ) override;

        private:
            compute::context context;
            compute::command_queue queue;
            size_t memory;
            ulong niter;
            ulong itersamples;
            uint seed;


            std::vector<Probability> accumulate_cpt(
                    std::vector<Probability> flat_cpt,
                    bn::state_t possible_states
            );

            std::vector<ulong> compute_result_general(bcvec &res);

            void calculate_iterations(const bayesian_network<Probability> &bn);

        };

        template<typename Probability>
        bn::marginal_distribution<Probability> logic_sampling<Probability>::make_inference(
                const bn::bayesian_network<Probability> &bn
        )
        {
            BAYLIB_ASSERT(std::all_of(bn.begin(), bn.end(),
                                      [](auto &var){ return bn::cpt_filled_out(var); }),
                          "conditional probability tables must be properly filled to"
                          " run logic_sampling inference algorithm",
                          std::runtime_error)

            calculate_iterations(bn);
            auto vertex_queue = bn::sampling_order(bn);
            compute::default_random_engine rand_eng = compute::default_random_engine (queue, seed);
            marginal_distribution<Probability> marginal_result(bn.begin(), bn.end());
            for (ulong i = 0; i< niter; i++) {
                uncompressed_result<Probability> result_container(vertex_queue.size());
                marginal_distribution<Probability> temp(bn.begin(), bn.end());
                //validity = compute::vector<bool>(itersamples, context);

                for(bn::vertex<Probability> v : vertex_queue) {

                    std::vector<bcvec> parents_result;
                    auto parents = bn[v].parents_info.names();
                    std::reverse(parents.begin(), parents.end());

                    for (auto p : parents) {
                        parents_result.push_back(result_container.get(bn.index_of(p)));
                    }

                    auto res = simulate_node( bn[v],
                                              parents_result,
                                              itersamples,
                                              rand_eng);

                    for(int ix=0; ix<res.first.size(); ix++)
                        marginal_result[v][ix] += res.first[ix];

                    result_container.put(res.second, v, bn.children_of(v).size());
                }
            }
            marginal_result.normalize();
            return marginal_result;
        }


        template <typename Probability>
        std::vector<Probability> logic_sampling<Probability>::accumulate_cpt(
                std::vector<Probability> flat_cpt,
                bn::state_t possible_states
        )
        {
            for(bn::state_t i = 0 ; i < flat_cpt.size() ; i += possible_states)
                for(bn::state_t j = 1 ; j < possible_states - 1 ; j++)
                    flat_cpt[i + j] += flat_cpt[i + j - 1];
            return flat_cpt;
        }

        /**
         * Accumulate simulation results for general case
         * @param res result from simulate node
         * @return vector for witch the i-th element is the number of occurrences of i
         **/
        template <typename T>
        std::vector<ulong> logic_sampling<T>::compute_result_general(bcvec& res)
        {
            std::vector<ulong> acc_res(res.cardinality);
            for (bn::state_t i = 0; i < res.cardinality; ++i) {
                acc_res[i] = compute::count(res.vec.begin(), res.vec.end(), i, queue);
            }
            return acc_res;
        }

        /**
         * Node Simulation with GPU parallelization
         * memory usage in GPU device: dim * (sizeof(Probability) + 3 * sizeof(ushort))) + cpt_len * sizeof(Probability)
         * @param flat_cpt CPT table in a contiguous format
         * @param parents_result output of parent nodes, if simulating source leave empty
         * @param dim number of samples to simulate, it must be consistent with parents simulation
         * @return pair of result vector and accumulated result of simulation
         * @param possible_states cardinality of the discrete variable to simulate (e.g. 2 if binary variable)
         **/
        template <typename Probability>
        std::pair<std::vector<ulong>, bcvec> logic_sampling<Probability>::simulate_node(
                const bn::random_variable<Probability> &var,
                const std::vector<bcvec>& parents_result,
                int dim,
                compute::default_random_engine rand_eng
        )
        {
            std::vector<Probability> flat_cpt_accum = accumulate_cpt(var.table().flat(), var.states().size());
            bcvec result = bcvec(dim, context, queue, var.states().size());
            prob_v device_cpt(flat_cpt_accum.size(), context);
            prob_v threshold_vec(dim, context);
            prob_v random_vec(dim, context);
            compute::uniform_real_distribution<Probability> distribution(0, 1);
            compute::vector<int> index_vec(dim, context);
            compute::copy(flat_cpt_accum.begin(), flat_cpt_accum.end(), device_cpt.begin(), queue);

            if(parents_result.empty()){
                compute::fill(index_vec.begin(), index_vec.end(), 0, queue);
            }
            else
            {
                int coeff = var.states().size();
                for (int i = 0; i < parents_result.size(); i++) {
                    if (i == 0)
                        compute::transform(parents_result[i].vec.begin(), parents_result[i].vec.end(),
                                           index_vec.begin(), _1 * coeff, queue);
                    else
                        compute::transform(parents_result[i].vec.begin(), parents_result[i].vec.end(),
                                           index_vec.begin(),index_vec.begin(),
                                           _1 * coeff + _2, queue);
                    coeff *= parents_result[i].cardinality;
                }
            }

            compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);

            distribution.generate(random_vec.begin(), random_vec.end(), rand_eng, queue);
            compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), result.vec.begin(), _1 > _2, queue);
            for(int i = 0; i + 2 < var.states().size() ; i++){
                compute::vector<int> temp(dim, context);
                compute::transform(index_vec.begin(), index_vec.end(),
                                   index_vec.begin(), _1 + 1, queue);
                compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(),
                                threshold_vec.begin(), queue);
                compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(),
                                   temp.begin(), _1 > _2, queue);
                compute::transform(temp.begin(), temp.end(), result.vec.begin(),
                                   result.vec.begin(), _1 + _2, queue);
            }
            std::vector<ulong> compr_res = compute_result_general(result);
            return {compr_res, result};
        }

        template<typename Probability>
        void logic_sampling<Probability>::calculate_iterations(const bayesian_network<Probability> &bn)
        {
            uint64_t sample_p = memory / (bn.number_of_variables() * sizeof(Probability) + 3 * sizeof(cl_ushort)) * MEMORY_SLACK;
            if(sample_p < this->nsamples){
                itersamples = sample_p;
                niter = this->nsamples / sample_p;
            }
            else
            {
                itersamples = this->nsamples;
                niter = 1;
            }
        }

    } // namespace inference
} // namespace bn


#endif //BAYLIB_LOGIC_SAMPLING_HPP