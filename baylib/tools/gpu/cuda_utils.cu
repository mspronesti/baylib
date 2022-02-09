//
// Created by paolo on 08/02/22.
//
#include "cuda_utils.cuh"

namespace baylib {

    /**
     * Kernel for accumulating multiple arrays into number of Blocks arrays
     * @param arr       : input array, will be overwritten by the result
     * @param var_num   : number of elements in a single array
     * @param n_set     : number of arrays
     */
    __global__ void reduce_marginal_array_kernel(uint *arr, uint var_num, uint n_set) {
        extern __shared__ uint s_data[];
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
     * Accumulate array of multiple marginal probability arrays into 1 single marginal vector
     * @param arr       : array of marginal array
     * @param var_num   : number of variables in the array
     * @param set_num   : number of arrays
     * @return          : marginal vector
     */
    std::vector<uint> reduce_marginal_array(uint *arr, uint var_num, uint set_num) {
        int max_shared_mem;
        size_t memory_for_single = var_num * sizeof(uint);
        int device;
        ulong chucks;
        gpuErrcheck(cudaGetDevice(&device));
        gpuErrcheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, device));

        uint thread_capacity = max_shared_mem / memory_for_single;
        size_t shared_mem_size;
        uint thread_num;
        if (thread_capacity > set_num) {
            chucks = 1;
            thread_num = set_num;
        } else {
            chucks = set_num / thread_capacity + 1;
            thread_num = thread_capacity;
        }
        shared_mem_size = var_num * thread_num * sizeof(uint);
        reduce_marginal_array_kernel<<<chucks, thread_num, shared_mem_size >>>(arr, var_num, set_num);

        std::vector<uint> marginal(var_num * chucks, 0);
        cudaMemcpy(marginal.data(), arr, sizeof(uint) * var_num * chucks, cudaMemcpyDeviceToHost);

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

}

