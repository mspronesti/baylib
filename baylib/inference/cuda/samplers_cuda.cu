//
// Created by paolo on 29/12/21.
//

#include "samplers_cuda.cuh"
#include <curand_kernel.h>
#include <baylib/tools/gpu/cuda_utils.cuh>
#include <baylib/tools/gpu/cuda_graph_adapter.cuh>

namespace inf = baylib::inference;
template std::vector<uint> inf::logic_sampler(
        cuda_graph_adapter<float> &graph, const std::vector<ulong>& sim_order, uint samples, bool evidence, size_t seed);
template std::vector<uint> inf::logic_sampler(
        cuda_graph_adapter<double> &graph, const std::vector<ulong>& sim_order, uint samples, bool evidence, size_t seed);
template std::vector<float> inf::likelihood_weighting_sampler(
        baylib::cuda_graph_adapter<float> &graph, const std::vector<ulong>& sim_order, uint samples, size_t seed);
template std::vector<float> inf::likelihood_weighting_sampler(
        baylib::cuda_graph_adapter<double> &graph, const std::vector<ulong>& sim_order, uint samples, size_t seed);


template<typename Probability>
__device__ uint discrete_sample(const Probability *distrib, uint size, curandState *state) {
    auto sample = static_cast<Probability>(curand_uniform(state));
    uint i = 0;
    Probability prob = distrib[0];
    while (sample > prob && i < size)
        prob += distrib[++i];
    return i;
}

namespace baylib {
    namespace inference {
        /**
         * Kernel for logic sampling simulation
         * @tparam Probability      : Probability type
         * @param graph             : cuda_variable vector that represents the whole network
         * @param sim_order         : Order of simulation
         * @param n_vars            : Number of variables in the graph (or length of graph vector)
         * @param marginal_result   : Array for storing results
         * @param total_states      : Overall number of states
         * @param n_iter            : Number of iterations
         * @param state             : Random state array for sample generation
         */
        template<bool check_evidence=true, typename Probability>
        __global__ void
        simulate_logic_kernel(const baylib::device_graph<Probability> graph,
                              const ushort *sim_order,
                              uint n_vars,
                              uint *marginal_result,
                              uint total_states,
                              uint n_iter,
                              curandState *state) {
            // Shared memory layout is as follows
            // each thread has a memory pool divided in two sections:
            // 1. section of n_vars length that holds the current state of the network during the simulation
            // 2. section of total_states length that holds the results for the simulations
            extern __shared__ ushort dynamic_mem_ls[];

            ushort *network_cache = &dynamic_mem_ls[(n_vars + total_states) * threadIdx.x];
            ushort *result_cache = &network_cache[n_vars];
            uint id = threadIdx.x + blockIdx.x * blockDim.x;
            curandState *local_state = &state[id];
            uint running_size;

            // Initialize 2 section of the cache
            for (int i = 0; i < total_states; i++)
                result_cache[i] = 0;

            for (int iter = 0; iter < n_iter; iter++) {
                bool valid_sample = true;
                running_size = 0;
                for (int i = 0; i < n_vars; i++) {
                    uint var_result;
                    ulong index = sim_order[i];
                    auto *cpt = reinterpret_cast<const Probability *>(graph.get_cpt(index));
                    ushort n_parents = graph.get_num_parents(index);
                    const ushort *parents = graph.get_parents(index);
                    ushort var_states = graph.get_n_states(index);
                    if (n_parents == 0) {
                        var_result = discrete_sample(cpt, var_states, local_state);
                    } else {
                        uint cpt_index = 0;
                        uint running_cpt_size = 1;
                        for (ushort j = 0; j < n_parents; j++) {
                            uint parent = parents[j];
                            cpt_index += network_cache[parent] * running_cpt_size * var_states;
                            running_cpt_size *= graph.get_n_states(parent);
                        }
                        var_result = discrete_sample(cpt + cpt_index, var_states, local_state);
                    }
                    if (check_evidence &&
                        graph.is_evidence(index) &&
                        graph.evidence_state(index) != var_result) {
                        valid_sample = false;
                        break;
                    }
                    network_cache[index] = var_result;
                    running_size += var_states;
                }
                if (valid_sample) {
                    running_size = 0;
                    for (int i = 0; i < n_vars; i++) {
                        ulong index = sim_order[i];
                        ushort var_states = graph.get_n_states(index);
                        result_cache[running_size + network_cache[index]] += 1;
                        running_size += var_states;
                    }
                }
            }

            // REDUCTION PHASE
            __syncthreads();

            uint step = total_states + n_vars;
            // Sum all results from a single block into a single marginal array
            for (uint s = 1; s < blockDim.x; s *= 2) {
                for (uint i = 0; i < total_states; ++i) {
                    uint ix = 2 * s * threadIdx.x;
                    if (ix + s < blockDim.x) {
                        dynamic_mem_ls[ix * step + i + n_vars] += dynamic_mem_ls[(ix + s) * step + i + n_vars];
                    }
                }
                __syncthreads();
            }

            // Copy the marginal array inside the cache to the result array
            if (threadIdx.x == 0) {
                for (int i = 0; i < total_states; i++) {
                    marginal_result[blockIdx.x * total_states + i] = dynamic_mem_ls[n_vars + i];
                }
            }
        }


        /**
         * Sampling for a bayesian network using cuda optimization
         * @tparam Probability : Type of the cpt entries
         * @param graph        : Bayesian Network
         * @param sim_order    : Order of simulation
         * @param samples      : Number of samples (The algorithm can generate more if needed for optimization purpouse)
         * @return             : Marginal array
         */
        template<typename Probability>
        std::vector<uint>
        logic_sampler(cuda_graph_adapter <Probability> &graph,
                      const std::vector<ulong> &sim_order,
                      uint samples,
                      bool evidence,
                      size_t seed) {
            ushort *sim_order_device;
            uint *marginal_result;
            curandState *curand_seeds;
            std::vector<ushort> sim_order_u(sim_order.begin(), sim_order.end());
            uint tot_states = 0;
            for (const baylib::cuda_variable<Probability> &var: graph.host_variables)
                tot_states += var.states;

            size_t single_shared_memory_size = sizeof(ushort) * (graph.total_dim + tot_states);
            baylib::kernel_params kp = calc_kernel_parameters(samples, single_shared_memory_size);
            gpuErrcheck(cudaMalloc(&sim_order_device, sim_order_u.size() * sizeof(uint)));
            gpuErrcheck(cudaMemcpy(sim_order_device, sim_order_u.data(), sim_order_u.size() * sizeof(ushort),
                                   cudaMemcpyHostToDevice));

            gpuErrcheck(cudaMalloc(&marginal_result, sizeof(uint) * tot_states * kp.N_Blocks));
            gpuErrcheck(cudaMalloc(&curand_seeds, kp.N_Blocks * kp.N_Threads * sizeof(curandState)));
            gpuErrcheck(cudaMemset(marginal_result, 0, sizeof(uint) * tot_states * kp.N_Blocks));

            baylib::setup_curand_kernel(curand_seeds, kp.N_Blocks * kp.N_Threads, seed);
            if(evidence) {
                simulate_logic_kernel<true><<<kp.N_Blocks, kp.N_Threads, single_shared_memory_size * kp.N_Threads>>>(
                        graph.load_graph_to_device(),
                        sim_order_device,
                        graph.total_dim,
                        marginal_result,
                        tot_states,
                        kp.N_Iter,
                        curand_seeds);
            }
            else{
                simulate_logic_kernel<false><<<kp.N_Blocks, kp.N_Threads, single_shared_memory_size * kp.N_Threads>>>(
                        graph.load_graph_to_device(),
                        sim_order_device,
                        graph.total_dim,
                        marginal_result,
                        tot_states,
                        kp.N_Iter,
                        curand_seeds);
            }

            auto marginal = baylib::reduce_marginal_array(marginal_result, tot_states, kp.N_Blocks);

            gpuErrcheck(cudaFree(sim_order_device));
            gpuErrcheck(cudaFree(curand_seeds));
            gpuErrcheck(cudaFree(marginal_result));
            cudaDeviceSynchronize();
            return marginal;
        }

        /**
         * Kernel for logic sampling simulation
         * @tparam Probability      : Probability type
         * @param graph             : cuda_variable vector that represents the whole network
         * @param sim_order         : Order of simulation
         * @param n_vars            : Number of variables in the graph (or length of graph vector)
         * @param marginal_result   : Array for storing results
         * @param total_states      : Overall numbenr of states
         * @param state             : Random state array for sample generation
         */
        template<typename Probability>
        __global__ void
        simulate_likelihood_kernel(baylib::device_graph<Probability> graph,
                                   const ushort *sim_order,
                                   float *marginal_result,
                                   uint total_states,
                                   const ushort *running_size_var,
                                   curandState *state) {
            // Shared memory layout is as follows
            // each thread has a memory pool divided in only one sections:
            // 1. a section of total_states length that holds the results for the simulation
            extern __shared__ float dynamic_mem_adaptive[];

            uint id = threadIdx.x + blockIdx.x * blockDim.x;
            uint tid = threadIdx.x;

            float *result_cache = &(dynamic_mem_adaptive[total_states * tid]);
            curandState *local_state = &state[id];
            uint running_size = 0;
            float likelihood = 1.0;

            // Initialize cache
            for (int i = 0; i < total_states; i++)
                result_cache[i] = 0.;

            for (int i = 0; i < graph.n_vars; i++) {
                uint var_result;
                ulong index = sim_order[i];

                auto *cpt = reinterpret_cast<const Probability *>(graph.get_cpt(index));
                ushort n_parents = graph.get_num_parents(index);
                const ushort *parents = graph.get_parents(index);
                ushort var_states = graph.get_n_states(index);
                if (n_parents == 0) {
                    var_result = discrete_sample(cpt, var_states, local_state);
                } else {
                    uint cpt_index = 0;
                    uint running_cpt_size = 1;
                    for (ushort j = 0; j < n_parents; j++) {
                        uint parent = parents[j];
                        uint parent_state = 0;
                        uint parent_running_size = running_size_var[parent];
                        for (; result_cache[parent_running_size + parent_state] == 0.0; parent_state++);
                        cpt_index += parent_state * running_cpt_size * var_states;
                        running_cpt_size *= graph.get_n_states(parent);
                    }

                    // If evidence node copy the evidence state and record the likelihood of the sample
                    if (graph.is_evidence(index)) {
                        var_result = graph.evidence_state(index);
                        likelihood *= (cpt + cpt_index)[var_result];
                    } else {
                        var_result = discrete_sample(cpt + cpt_index, var_states, local_state);
                    }
                }
                result_cache[running_size + var_result] = 1.0;
                running_size += var_states;
            }

            for (int i = 0; i < total_states; i++) {
                result_cache[i] *= likelihood;
            }

            // REDUCTION PHASE
            __syncthreads();
            // Sum all results from a single block into a single marginal array
            for (uint s = 1; s < blockDim.x; s *= 2) {
                uint ix = 2 * s * tid;
                uint step = total_states;
                if (ix + s < blockDim.x) {
                    for (uint i = 0; i < total_states; i++) {
                        dynamic_mem_adaptive[ix * step + i] += dynamic_mem_adaptive[(ix + s) * step + i];
                    }
                }
                __syncthreads();
            }

            // Copy the marginal array inside the cache to the result array
            float *marginal_write = &marginal_result[blockIdx.x * total_states];
            for (int i = 0; i < total_states / blockDim.x + 1; i++) {
                if (blockDim.x * i + tid < total_states) {
                    marginal_write[blockDim.x * i + tid] = dynamic_mem_adaptive[blockDim.x * i + tid];
                }
            }
        }


        /**
         * likelihood weighting sampler, the marginal distribution is obtained by sampling the
         * network and weighting the result by
         * @tparam Probability Type of the cpt entries
         * @param graph        : Bayesian Network
         * @param sim_order    : Order of simulation
         * @param samples      : Number of samples (The algorithm can generate more if needed for optimization purpouse)
         * @return             : Marginal array
         */
        template<typename Probability>
        std::vector<float>
        likelihood_weighting_sampler(baylib::cuda_graph_adapter<Probability> &graph,
                                     const std::vector<ulong> &sim_order,
                                     uint samples,
                                     size_t seed) {
            ushort *sim_order_device;
            float *marginal_result_device;
            ushort *running_size_device;
            curandState *curand_seeds;
            std::vector<ushort> sim_order_u(sim_order.begin(), sim_order.end());
            std::vector<ushort> running_size(sim_order.size(), 0);
            uint tot_states = 0;
            for (int i = 0; i < graph.host_variables.size(); i++) {
                running_size[sim_order[i]] = tot_states;
                tot_states += graph.host_variables[sim_order[i]].states;
            }

            size_t single_shared_memory_size = sizeof(float) * tot_states;
            baylib::kernel_params kp = calc_kernel_parameters(samples, single_shared_memory_size);
            gpuErrcheck(cudaMalloc(&sim_order_device, sim_order_u.size() * sizeof(ushort)));
            gpuErrcheck(cudaMalloc(&running_size_device, running_size.size() * sizeof(ushort)));
            gpuErrcheck(cudaMemcpy(sim_order_device, sim_order_u.data(), sim_order_u.size() * sizeof(ushort),
                                   cudaMemcpyHostToDevice));
            gpuErrcheck(cudaMemcpy(running_size_device, running_size.data(), running_size.size() * sizeof(ushort),
                                   cudaMemcpyHostToDevice));
            gpuErrcheck(cudaMalloc(&marginal_result_device, sizeof(float) * tot_states * kp.N_Blocks));
            gpuErrcheck(cudaMalloc(&curand_seeds, kp.N_Blocks * kp.N_Threads * sizeof(curandState)))
            gpuErrcheck(cudaMemset(marginal_result_device, 0, sizeof(float) * tot_states * kp.N_Blocks));

            baylib::setup_curand_kernel(curand_seeds, kp.N_Blocks * kp.N_Threads, seed);
            simulate_likelihood_kernel<<<kp.N_Blocks, kp.N_Threads, single_shared_memory_size * kp.N_Threads>>>(
                    graph.load_graph_to_device(),
                    sim_order_device,
                    marginal_result_device,
                    tot_states,
                    running_size_device,
                    curand_seeds);
            auto marginal = baylib::reduce_marginal_array(marginal_result_device, tot_states, kp.N_Blocks);
            gpuErrcheck(cudaFree(sim_order_device));
            gpuErrcheck(cudaFree(curand_seeds));
            gpuErrcheck(cudaFree(marginal_result_device));
            gpuErrcheck(cudaFree(running_size_device));
            cudaDeviceSynchronize();
            return marginal;
        }
    }
}
