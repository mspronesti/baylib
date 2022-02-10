//
// Created by paolo on 29/12/21.
//

#include "logic_sampler.cuh"
#include <curand_kernel.h>
#include <baylib/tools/gpu/cuda_utils.cuh>

template std::vector<uint> baylib::inference::simulate(baylib::cuda_graph<float>& graph, const std::vector<ulong>& sim_order, uint samples);
template std::vector<uint> baylib::inference::simulate(baylib::cuda_graph<double>& graph, const std::vector<ulong>& sim_order, uint samples);


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
        template<typename Probability>
        __global__ void
        simulate_kernel(const baylib::cuda_variable<Probability> *graph,
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
            extern __shared__ ushort dynamic_mem[];

            ushort *network_cache = &dynamic_mem[(n_vars + total_states) * threadIdx.x];
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
                    auto *cpt = reinterpret_cast<Probability *>(graph[index].data);
                    ushort n_parents = graph[index].number_parents;
                    ushort *parents = graph[index].device_parents;
                    ushort var_states = graph[index].states;
                    if (n_parents == 0) {
                        var_result = baylib::discrete_sample(cpt, var_states, local_state);
                    } else {
                        uint cpt_index = 0;
                        uint running_cpt_size = 1;
                        for (ushort j = 0; j < n_parents; j++) {
                            uint parent = parents[j];
                            cpt_index += network_cache[parent] * running_cpt_size * var_states;
                            running_cpt_size *= graph[parent].states;
                        }
                        var_result = baylib::discrete_sample(cpt + cpt_index, var_states, local_state);
                    }
                    if(graph[index].is_evidence && graph[index].evidence_state != var_result){
                        valid_sample = false;
                        break;
                    }
                    network_cache[index] = var_result;
                    running_size += var_states;
                }
                if(valid_sample) {
                    running_size = 0;
                    for (int i = 0; i < n_vars; i++) {
                        ulong index = sim_order[i];
                        ushort var_states = graph[index].states;
                        result_cache[running_size + network_cache[index]] += 1;
                        running_size += var_states;
                    }
                }
            }

            // REDUCTION PHASE
            __syncthreads();

            // Sum all results from a single block into a single marginal array
            for (uint s = 1; s < blockDim.x; s *= 2) {
                for (uint i = 0; i < total_states; ++i) {
                    uint ix = 2 * s * threadIdx.x;
                    uint step = total_states + n_vars;
                    if (ix + s < blockDim.x) {
                        dynamic_mem[ix * step + i + n_vars] += dynamic_mem[(ix + s) * step + i + n_vars];
                    }
                }
                __syncthreads();
            }

            // Copy the marginal array inside the cache to the result array
            if (threadIdx.x == 0) {
                for (int i = 0; i < total_states; i++) {
                    marginal_result[blockIdx.x * total_states + i] = dynamic_mem[n_vars + i];
                }
            }

        }

        /**
         * Calculate kernel dimensions depending on the needed memory
         * @param samples               : Number of samples requested
         * @param shared_mem_per_thread : Memory needed by every single thread
         * @return                      : kernel parameters
         */
        baylib::kernel_params calc_kernel_parameters(uint samples, size_t shared_mem_per_thread) {
            int max_shared_mem;
            int device;
            baylib::kernel_params result{};
            int max_launchable_threads;
            int max_launchable_blocks;
            int warp_size;
            uint max_threads;
            uint n_threads;
            uint n_blocks;
            uint n_iter;
            gpuErrcheck(cudaGetDevice(&device));
            gpuErrcheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, device));
            gpuErrcheck(cudaDeviceGetAttribute(&max_launchable_threads, cudaDevAttrMaxThreadsPerBlock, device));
            gpuErrcheck(cudaDeviceGetAttribute(&max_launchable_blocks, cudaDevAttrMaxGridDimX, device));
            gpuErrcheck(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
            max_threads = max_shared_mem / shared_mem_per_thread;
            n_threads = max_threads < max_launchable_threads ? max_threads : max_launchable_threads;
            if (n_threads > warp_size)
                n_threads -= n_threads % warp_size;
            result.N_Threads = n_threads;
            n_blocks = samples / n_threads + 1;
            n_blocks = n_blocks < max_launchable_blocks ? n_blocks : max_launchable_blocks;
            result.N_Blocks = n_blocks;
            n_iter = samples / (n_threads * n_blocks) + 1;
            result.N_Iter = n_iter;
            return result;
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
        simulate(baylib::cuda_graph<Probability> &graph, const std::vector<ulong> &sim_order, uint samples) {
            ushort *sim_order_device;
            baylib::cuda_variable<Probability> *graph_device;
            uint *marginal_result;
            curandState *devStates;
            std::vector<ushort> sim_order_u(sim_order.begin(), sim_order.end());
            uint tot_states = 0;
            for (const baylib::cuda_variable<Probability> &var: graph.host_variables)
                tot_states += var.states;

            size_t single_shared_memory_size = sizeof(ushort) * (graph.total_dim + tot_states);
            baylib::kernel_params kp = calc_kernel_parameters(samples, single_shared_memory_size);
            gpuErrcheck(cudaMalloc(&sim_order_device, sim_order_u.size() * sizeof(uint)));
            gpuErrcheck(cudaMemcpy(sim_order_device, sim_order_u.data(), sim_order_u.size() * sizeof(ushort),
                                   cudaMemcpyHostToDevice));
            gpuErrcheck(cudaMalloc(&graph_device, sizeof(baylib::cuda_variable<Probability>) * graph.total_dim));
            gpuErrcheck(cudaMalloc(&marginal_result, sizeof(uint) * tot_states * kp.N_Blocks));
            gpuErrcheck(cudaMalloc(&devStates, kp.N_Blocks * kp.N_Threads * sizeof(curandState)))
            gpuErrcheck(cudaMemset(marginal_result, 0, sizeof(uint) * tot_states * kp.N_Blocks));
            gpuErrcheck(cudaMemcpy(graph_device, graph.host_variables.data(),
                                   sizeof(baylib::cuda_variable<Probability>) * graph.total_dim,
                                   cudaMemcpyHostToDevice));

            baylib::setup_kernel<<<kp.N_Blocks, kp.N_Threads>>>(devStates);
            simulate_kernel<<<kp.N_Blocks, kp.N_Threads, single_shared_memory_size * kp.N_Threads>>>(
                    graph_device,
                    sim_order_device,
                    graph.total_dim,
                    marginal_result,
                    tot_states,
                    kp.N_Iter,
                    devStates);
            auto marginal = baylib::reduce_marginal_array(marginal_result, tot_states, kp.N_Blocks);
            gpuErrcheck(cudaFree(sim_order_device));
            gpuErrcheck(cudaFree(graph_device));
            gpuErrcheck(cudaFree(devStates));
            gpuErrcheck(cudaFree(marginal_result));
            cudaDeviceSynchronize();
            return marginal;
        }
    }
}
