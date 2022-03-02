//
// Created by paolo on 22/12/21.
//

#ifndef BAYLIB_CUDA_UTILS_CUH
#define BAYLIB_CUDA_UTILS_CUH


#include <vector>
#include <cstdio>
#include <curand_kernel.h>

#define gpuErrcheck(ans) { baylib::gpuAssert((ans), __FILE__, __LINE__); }

namespace baylib {

    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }


    /**
     * Struct for holding the launch parameters of a kernel
     */
    struct kernel_params {
        uint N_Blocks;
        uint N_Threads;
        uint N_Iter;
    };



    /**
     * Accumulate array of multiple marginal probability arrays into 1 single marginal vector
     * @param arr       : array of marginal array
     * @param var_num   : number of variables in the array
     * @param set_num   : number of arrays
     * @return          : marginal vector
     */
    template<typename T>
    std::vector<T> reduce_marginal_array(T *arr, uint var_num, uint set_num);

    /**
     * Setup curandState for curand library
     * @param state : output array of dimension equal to the number of launched threads
     */
    void setup_curand_kernel(curandState *state, uint length, size_t seed=7);

    /**
    * Calculate kernel dimensions depending on the needed memory
    * @param samples               : Number of samples requested
    * @param shared_mem_per_thread : Memory needed by every single thread
    * @return                      : kernel parameters
    */
    baylib::kernel_params calc_kernel_parameters(uint samples, size_t shared_mem_per_thread);

}
#endif //BAYLIB_CUDA_UTILS_CUH
