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
     * Struct for managing a single variable of a bayesian network and its related resources on gpu
     * @tparam probability_type
     */
    template<typename probability_type>
    struct cuda_variable {
        probability_type *data;
        ushort *device_parents{};
        ushort number_parents;
        ushort states;
        bool is_evidence;
        ushort evidence_state;

        /**
         * cuda_variable constructor
         */
        explicit cuda_variable() :
                data(nullptr),
                number_parents(0),
                states(0),
                is_evidence(false),
                device_parents(nullptr),
                evidence_state(0) {
        }

        /**
         * fill variable with relevant data
         * @param n_states      : number of states
         * @param n_parents     : number of parents
         * @param n_conditions  : number of conditions of the cpt
         */
        void fill(ushort n_states, ushort n_parents, ushort n_conditions) {
            states = n_states;
            number_parents = n_parents;
            is_evidence = false;
            evidence_state = 0;
            cudaMalloc(&data, n_conditions * n_states * sizeof(probability_type));
            cudaMalloc(&device_parents, n_parents * sizeof(ushort));
        }

        /**
         * Set the node to evidence node
         * @param evidence  : observed state
         */
        void set_evidence(ushort evidence) {
            evidence_state = evidence;
            is_evidence = true;
        }

        cuda_variable(cuda_variable<probability_type> &) = delete;

        cuda_variable(cuda_variable<probability_type> &&) noexcept = default;

        cuda_variable<probability_type> &operator=(cuda_variable<probability_type> &&) noexcept = default;


        ~cuda_variable() {
            if (data != nullptr) {
                cudaFree(data);
                data = nullptr;
            }
            if (device_parents != nullptr) {
                cudaFree(device_parents);
                device_parents = nullptr;
            }
        }
    };

    /**
     * Bayesian network compatible with CUDA algorithms, contains only the bare minimum information
     * in order to optimize performance
     * @tparam Probability_ : type of probability, can only be float or double
     */
    template<typename Probability_>
    struct cuda_graph {
        std::vector<cuda_variable<Probability_>> host_variables;
        ulong total_dim{};

        /**
         * cuda_graph constructor
         * @param n_variables   : number of variables in the graph
         */
        explicit cuda_graph(ulong n_variables) : total_dim(n_variables), host_variables(n_variables) {}

        /**
         * Add variable contents
         * @param index         : index of variable
         * @param cpt           : unrolled cpt in the form of vector
         * @param n_states      : number of possible states for the variable
         * @param n_parents     : number of parents
         * @param n_conditions  : number of conditions in the cpt
         */
        void
        add_variable(ulong index, std::vector<Probability_> cpt, ulong n_states, ulong n_parents, ulong n_conditions) {
            host_variables[index].fill(n_states, n_parents, n_conditions);
            gpuErrcheck(cudaMemcpy(host_variables[index].data, cpt.data(), sizeof(Probability_) * cpt.size(),
                                   cudaMemcpyHostToDevice));
        }

        /**
         * Add graph edges
         * @param index_children    : index of the chile
         * @param parents           : vector of indices of the parents
         */
        void add_dependencies(ulong index_children, const std::vector<ulong> &parents) {
            if (!parents.empty()) {
                std::vector<ushort> parents_s(parents.begin(), parents.end());
                size_t arr_size = parents.size() * sizeof(ushort);
                gpuErrcheck(cudaMemcpy(host_variables[index_children].device_parents, parents_s.data(), arr_size,
                                       cudaMemcpyHostToDevice));
            }
        }

        cuda_graph(const cuda_graph<Probability_> &) = delete;

        cuda_graph(cuda_graph<Probability_> &&) noexcept = default;

        cuda_graph<Probability_> &operator=(cuda_graph<Probability_> &&) noexcept = default;

    };

    /**
     * Struct for holding the launch parameters of a kernel
     */
    struct kernel_params {
        uint N_Blocks;
        uint N_Threads;
        uint N_Iter;
    };

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
    std::vector<uint> reduce_marginal_array(uint *arr, uint var_num, uint set_num);

    /**
     * Setup curandState for curand library
     * @param state : output array of dimension equal to the number of launched threads
     */
    __global__ void setup_kernel(curandState *state);

}
#endif //BAYLIB_CUDA_UTILS_CUH
