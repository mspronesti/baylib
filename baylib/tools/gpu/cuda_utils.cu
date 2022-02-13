//
// Created by paolo on 08/02/22.
//
#include "cuda_utils.cuh"

namespace baylib {

    template std::vector<uint> reduce_marginal_array(uint *arr, uint var_num, uint set_num);
    template std::vector<float> reduce_marginal_array(float *arr, uint var_num, uint set_num);

    /**
     * Kernel for accumulating multiple arrays into number of Blocks arrays
     * @param arr       : input array, will be overwritten by the result
     * @param var_num   : number of elements in a single array
     * @param n_set     : number of arrays
     */
     template <typename T>
    __global__ void reduce_marginal_array_kernel(T *arr, uint var_num, uint n_set) {
        extern __shared__ char shared_mem[];
        T* s_data = reinterpret_cast<T*>(shared_mem);
        uint id = threadIdx.x + blockDim.x * blockIdx.x;
        uint tid = threadIdx.x;

        // Load elements into shared memory
        for (uint i = 0; i < var_num; i++) {
            if (id < n_set)
                s_data[tid * var_num + i] = arr[id * var_num + i];
        }

        // Accumulate elemets into first array
        __syncthreads();
        for (uint s = 1; s < blockDim.x; s *= 2) {
            for (int i = 0; i < var_num; i++) {
                if (tid % (2 * s) == 0 && (tid + s) < blockDim.x && (id + s) < n_set) {
                    s_data[tid * var_num + i] += s_data[(tid + s) * var_num + i];
                }
            }
            __syncthreads();
        }

        // Copy array into the first elements of the output array
        if (tid == 0) {
            for (uint i = 0; i < var_num; i++) {
                arr[blockIdx.x * var_num + i] = s_data[i];
            }
        }

    }

    /**
     * Sample from a given discrete distribution while inside a cuda kernel
     * @tparam Probability  : Probability type of the distribution
     * @param distrib       : Distribution array
     * @param size          : size of the distribution
     * @param state         : CurandState for curand library
     * @return              : sample
     */
    template<typename Probability>
    __device__ uint discrete_sample(Probability *distrib, uint size, curandState *state) {
        auto sample = static_cast<Probability>(curand_uniform(state));
        uint i = 0;
        Probability prob = distrib[0];
        while (sample > prob && i < size)
            prob += distrib[++i];
        return i;
    }


    /**
     * Accumulate array of multiple marginal probability arrays into 1 single marginal vector
     * @param arr       : array of marginal array
     * @param var_num   : number of variables in the array
     * @param set_num   : number of arrays
     * @return          : marginal vector
     */
    template<typename T>
    std::vector<T> reduce_marginal_array(T *arr, uint var_num, uint set_num) {
        int max_shared_mem;
        size_t memory_for_single = var_num * sizeof(T);
        int device;
        int max_threads_per_block;
        ulong chucks;
        gpuErrcheck(cudaGetDevice(&device));
        gpuErrcheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, device));
        gpuErrcheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
        uint thread_capacity = max_shared_mem / memory_for_single;
        thread_capacity = thread_capacity > max_threads_per_block ? max_threads_per_block : thread_capacity;
        size_t shared_mem_size;
        chucks = set_num / thread_capacity + 1;

        shared_mem_size = var_num * thread_capacity * sizeof(T);
        reduce_marginal_array_kernel<<<chucks, thread_capacity, shared_mem_size >>>(arr, var_num, set_num);


        std::vector<T> marginal(var_num * set_num, 0);
        cudaMemcpy(marginal.data(), arr, sizeof(T) * var_num * set_num, cudaMemcpyDeviceToHost);

        if (chucks > 1) {
            for (int i = 1; i < chucks; i++) {
                for (int j = 0; j < var_num; j++) {
                    marginal[j] += marginal[i * var_num + j];
                }
            }
        }
        return marginal;
    }

    /**
     * Setup curandState for curand library
     * @param state : output array of dimension equal to the number of launched threads
     */
    __global__ void setup_kernel(curandState *state) {
        uint id = threadIdx.x + blockIdx.x * blockDim.x;
        /* Each thread gets different seed, a different sequence
           number, no offset */
        curand_init(7 + id, id, 0, &state[id]);
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

}

