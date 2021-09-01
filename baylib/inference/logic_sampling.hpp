//
// Created by elle on 22/07/21.
//

#ifndef BAYLIB_LOGIC_SAMPLING_HPP
#define BAYLIB_LOGIC_SAMPLING_HPP


#define MEMORY_SLACK 0.8

#include <vector>
#include <future>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>

#include <baylib/inference/inference_algorithm.h>

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
            data(std::vector<std::pair<std::vector<ulong>, std::optional<bcvec>>>(nnodes)),
            use_count(std::vector<ulong>(nnodes, 0))
            {}

            void put(const std::pair<std::vector<ulong>, std::optional<bcvec>>& entry, bn::vertex<Probability> edge, ulong uses) {
                data[edge] = entry;
                use_count[edge] = uses;
            }

            std::pair<std::vector<ulong>, std::optional<bcvec>> get(const bn::vertex<Probability> v) {
                auto res = data[v];
                BAYLIB_ASSERT(use_count[v] > 0,
                              "Function get has been called on node "
                              << v << " too many times",
                              std::runtime_error)

                use_count[v]--;
                if (use_count[v] == 0)
                    data[v].second.reset();
                return res;
            }

            std::vector<ulong>& get_compressed(const bn::vertex<Probability> v) {
                return data[v].first;
            }

        private:
            std::vector<std::pair<std::vector<ulong>, std::optional<bcvec>>> data;
            std::vector<ulong> use_count;

            std::pair<std::vector<ulong>,bcvec> & operator[] (const bn::vertex<Probability> v)
            {
                return data[v];
            }
        };

        /**
         * This class models the result in reduced form
         * (number of realizations of a specific state)
         * Can be added to other compressed_results.
         * Can add data from uncompressed_result
         * Used to compute marginal_distribution
         * @tparam Probability : the type modeling probability
         **/
        template <typename Probability>
        class compressed_result {
        private:
            std::vector<std::vector<ulong>> data;
            ulong nsamples;
            bn::vertex<Probability> nnodes;
        public:
            //nsamples is not assigned from the constructor,
            // it will get its value from uncompressed_results
            compressed_result(
                    ulong nsamples,
                    ulong nnodes
            )
            : nsamples(nsamples)
            , nnodes(nnodes)
            , data(std::vector<std::vector<ulong>>(nnodes) )
            { }

            compressed_result &operator=(const compressed_result &other) {
                if (this == &other)
                    return *this;
                other.data.swap(*this);
                return *this;
            }

            void swap(compressed_result &other) {
                data.swap(other.data);
            }

            compressed_result& operator+=(compressed_result &other) {
                BAYLIB_ASSERT(other.nsamples == nsamples,
                              "Samples vector size does not agree.",
                              std::runtime_error)

                BAYLIB_ASSERT(other.nnodes == nnodes,
                               "Edge number does not agree.",
                               std::runtime_error)

                for (bn::vertex<Probability> i = 0; i < nnodes; i++) {
                    for (bn::state_t j = 0; j<data[i].size(); j++) {
                        data[i][j] += other.data[i][j];
                    }
                }
                return (*this);
            }

            compressed_result operator + (compressed_result &other) {
                compressed_result new_spr = (*this);
                new_spr+=other;
                return new_spr;
            }

            std::vector<ulong>& operator[](const bn::vertex<Probability> key) {
                return data[key];
            }

            compressed_result& operator+=(uncompressed_result<Probability> &other) {
                for (bn::vertex<Probability> i = 0; i< data.size();i++) {
                    if (data[i].size() == 0) {
                        data[i] = std::vector<ulong>(other.get_compressed(i).size(), 0);
                    }
                    for (bn::state_t j = 0; j<data[i].size();j++) {
                        data[i][j] += other.get_compressed(i)[j];
                    }
                }
                return (*this);
            }

            ulong num_nodes() {
                return nnodes;
            }
        };

        /** ===== Logic Sampling Algorithm ===
        *
        * This class represents the Logic Sampling approximate
        * inference algorithm for discrete Bayesian Networks.
        * @tparam Probability : the type expressing the probability
        */
        template <typename Probability>
        class logic_sampling : inference_algorithm<Probability>{
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
                    std::vector<Probability> flat_cpt,
                    const std::vector<bcvec>& parents_result,
                    int dim,
                    int possible_states,
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
            uint seed;

            std::vector<Probability> accumulate_cpt(
                    std::vector<Probability> flat_cpt,
                    bn::state_t possible_states
            );

            std::vector<ulong> compute_result_general(bcvec &res);

            void calculate_iterations(const bayesian_network<Probability> &bn);

            bn::marginal_distribution<Probability> compute_total_result(
                    const bayesian_network<Probability> &bn,
                    compressed_result <Probability> &cr
            );
        };

        template<typename Probability>
        bn::marginal_distribution<Probability> logic_sampling<Probability>::make_inference(
                const bn::bayesian_network<Probability> &bn
        )
        {
            calculate_iterations(bn);
            auto var = bn.variables();
            BAYLIB_ASSERT(std::all_of(var.begin(), var.end(),
                                      [](auto &var){ return bn::cpt_filled_out(var); }),
                          "conditional probability tables must be properly filled to"
                          " run logic_sampling inference algorithm",
                          std::runtime_error)

            auto vertex_queue = bn::sampling_order(bn);
            compute::default_random_engine rand_eng = compute::default_random_engine (queue, seed);
            compressed_result<Probability> pr(this->nsamples, vertex_queue.size());

            for (ulong i = 0; i< niter; i++) {
                uncompressed_result<Probability> results(vertex_queue.size());

                for(bn::vertex<Probability> v : vertex_queue) {

                    std::vector<bcvec> parents_result;
                    auto parents = bn[v].parents_info.names();
                    std::reverse(parents.begin(), parents.end());

                    for (auto p : parents) {
                        std::pair<std::vector<ulong>, std::optional<bcvec>> par = results.get(bn.index_of(p));
                        bcvec par_res = par.second.value();
                        parents_result.push_back(par_res);
                    }

                    auto res = simulate_node( bn[v].table().flat(),
                                              parents_result,
                                              this->nsamples,
                                              bn[v].states().size(),
                                              rand_eng);
                    results.put(res, v, bn.children_of(v).size());
                }

                pr += results;
            }

            return compute_total_result(bn, pr);
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


        /// Accumulate simulation results for general case
        /// \param res result from simulate node
        /// \return vector for witch the i-th element is the number of occurrences of i
        template <typename T>
        std::vector<ulong> logic_sampling<T>::compute_result_general(bcvec& res)
        {
            std::vector<ulong> acc_res(res.cardinality);
            for (bn::state_t i = 0; i < res.cardinality; ++i) {
                acc_res[i] = compute::count(res.vec.begin(), res.vec.end(), i, queue);
            }
            return acc_res;
        }


        /// Node Simulation with GPU parallelization
        /// memory usage in GPU device: dim * (sizeof(Probability) + 3 * sizeof(ushort))) + cpt_len * sizeof(Probability)
        /// \param flat_cpt CPT table in a contiguous format
        /// \param parents_result output of parent nodes, if simulating source leave empty
        /// \param dim number of samples to simulate, it must be consistent with parents simulation
        /// \param possible_states cardinality of the discrete variable to simulate (e.g. 2 if binary variable)
        /// \return shared_ptr to result of simulation, use with other simulations or condense results with compute_result
        template <typename Probability>
        std::pair<std::vector<ulong>, bcvec> logic_sampling<Probability>::simulate_node(
                const std::vector<Probability> flat_cpt,
                const std::vector<bcvec>& parents_result,
                int dim,
                int possible_states,
                compute::default_random_engine rand_eng
        )
        {
            std::vector<Probability> flat_cpt_accum = accumulate_cpt(flat_cpt, possible_states);
            bcvec result = bcvec(dim, context, queue, possible_states);
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
                int coeff = possible_states;
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
            for(int i = 0; i + 2 < possible_states ; i++){
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
                this->nsamples = sample_p;
                niter = this->nsamples / sample_p;
            }
            else
            {
                niter = 1;
            }
        }

        template<typename Probability>
        bn::marginal_distribution<Probability> logic_sampling<Probability>::compute_total_result(
                const bayesian_network<Probability> &bn,
                compressed_result<Probability> &cr
        )
        {
            bn::marginal_distribution<Probability> total_result(bn.variables());
            for (bn::vertex<Probability> v = 0; v< cr.num_nodes(); v++) {
                for (bn::state_t s = 0; s < cr[v].size(); ++s)
                    total_result.set(v,s,Probability(cr[v][s])/(this->nsamples * niter));
            }
            return total_result;
        }
    } // namespace inference
} // namespace bayes_net


#endif //BAYLIB_LOGIC_SAMPLING_HPP