//
// Created by paolo on 14/02/22.
//

#ifndef BAYLIB_CUDA_GRAPH_revised_REVISED_CUH
#define BAYLIB_CUDA_GRAPH_revised_REVISED_CUH

#include <vector>
#include <cstdio>
#include <curand_kernel.h>

namespace baylib {


    /**
     * Struct for managing a single variable of a bayesian network and its related resources on gpu
     * @tparam probability_type
     */
    template<typename probability_type>
    struct cuda_variable {
        uint data_index;
        ushort parents_index;
        ushort number_parents;
        ushort states;
        bool is_evidence;
        ushort evidence_state;

        /**
         * cuda_variable constructor
         */
        explicit cuda_variable() :
                data_index(0),
                number_parents(0),
                states(0),
                is_evidence(false),
                parents_index(0),
                evidence_state(0) {
        }

        /**
         * fill variable with relevant data
         * @param n_states      : number of states
         * @param n_parents     : number of parents
         * @param index  : relative position in the cpt_array
         */
        void fill(ushort n_states, ushort n_parents, ushort index) {
            states = n_states;
            number_parents = n_parents;
            data_index = index;
            is_evidence = false;
            evidence_state = 0;
        }

        /**
         * Set the node to evidence node
         * @param evidence  : observed state
         */
        void set_evidence(ushort evidence) {
            evidence_state = evidence;
            is_evidence = true;
        }

    };

    /**
     * Bayesian network representation for cuda device, all elements live only in device memory
     * the lifespan of this struct is bounded by the cuda_graph_adapter that generated it
     * @tparam Probability_
     */
    template<typename Probability_>
    struct device_graph{
        cuda_variable<Probability_>* device_variables;
        Probability_* cpt_array_device;
        ushort* parents_array_device;
        ulong n_vars;

        __device__ const Probability_* get_cpt(uint index) const{
            return &cpt_array_device[device_variables[index].data_index];
        }

        __device__ const ushort* get_parents(uint index) const{
            return &parents_array_device[device_variables[index].parents_index];
        }

        __device__ ushort get_num_parents(uint index) const {
            return device_variables[index].number_parents;
        }

        __device__ ushort get_n_states(uint index) const{
            return device_variables[index].states;
        }

        __device__ bool is_evidence(uint index) const{
            return device_variables[index].is_evidence;
        }

        __device__ uint evidence_state(uint index) const{
            return device_variables[index].evidence_state;
        }
    };

/**
 * Adapter for creating and managing the resources for device graphs
 * @tparam Probability_ : type of probability, only float or double are supported by the kernels in the library
 */
    template<typename Probability_>
    struct cuda_graph_adapter {
        std::vector<cuda_variable<Probability_>> host_variables;
        std::vector<Probability_> cpt_array;
        std::vector<ushort> parents_array;
        Probability_* cpt_array_device;
        ushort* parents_array_device;
        cuda_variable<Probability_> *device_variables;
        ulong total_dim{};

        /**
         * cuda_graph_adapter constructor
         * @param n_variables   : number of variables in the graph
         */
        explicit cuda_graph_adapter(ulong n_variables) :
                total_dim(n_variables),
                host_variables(n_variables),
                device_variables(nullptr),
                cpt_array_device(nullptr),
                parents_array_device(nullptr){}

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
            host_variables[index].fill(n_states, n_parents, cpt_array.size());
            cpt_array.insert(cpt_array.end(), cpt.begin(), cpt.end());
            //gpuErrcheck(cudaMemcpy(host_variables[index].data, cpt.data(), sizeof(Probability_) * cpt.size(),
            //                       cudaMemcpyHostToDevice));
        }

        /**
         * Add graph edges
         * @param index_children    : index of the chile
         * @param parents           : vector of indices of the parents
         */
        void add_dependencies(ulong index_children, const std::vector<ulong> &parents) {
            if (!parents.empty()) {
                host_variables[index_children].parents_index = parents_array.size();
                parents_array.insert(parents_array.end(), parents.begin(), parents.end());

                //gpuErrcheck(cudaMemcpy(host_variables[index_children].device_parents, parents_s.data(), arr_size,
                //                       cudaMemcpyHostToDevice));
            }
        }

        /**
         * Load graph into gpu memory if not already loaded
         * @return true if loading was successful, false if there was already a loaded graph
         */
        device_graph<Probability_> load_graph_to_device() {
            device_graph<Probability_> graph;
            if (device_variables == nullptr) {
                gpuErrcheck(cudaMalloc(&device_variables, sizeof(baylib::cuda_variable<Probability_>) * total_dim));
                gpuErrcheck(cudaMemcpy(device_variables, host_variables.data(),
                                       sizeof(baylib::cuda_variable<Probability_>) * total_dim,
                                       cudaMemcpyHostToDevice));
                gpuErrcheck(cudaMalloc(&cpt_array_device, sizeof(Probability_) * cpt_array.size()));
                gpuErrcheck(cudaMemcpy(cpt_array_device, cpt_array.data(),
                                       sizeof(Probability_) * cpt_array.size(),
                                       cudaMemcpyHostToDevice));
                gpuErrcheck(cudaMalloc(&parents_array_device, sizeof(ushort) * parents_array.size()));
                gpuErrcheck(cudaMemcpy(parents_array_device, parents_array.data(),
                                       sizeof(ushort) * parents_array.size(),
                                       cudaMemcpyHostToDevice));
            }
            graph.cpt_array_device = cpt_array_device;
            graph.parents_array_device = parents_array_device;
            graph.device_variables = device_variables;
            graph.n_vars = total_dim;
            return graph;
        }


        /**
         * Set node to evidence node
         * @param index : index of the node
         * @param state : evidence state
         */
        void set_evidence(ulong index, ushort state) {
            host_variables[index].set_evidence(state);
        }

        cuda_graph_adapter(const cuda_graph_adapter<Probability_> &) = delete;

        cuda_graph_adapter(cuda_graph_adapter<Probability_> &&) noexcept = default;

        cuda_graph_adapter<Probability_> &operator=(cuda_graph_adapter<Probability_> &&) noexcept = default;

        ~cuda_graph_adapter<Probability_>() {
            if (device_variables != nullptr) {
                gpuErrcheck(cudaFree(device_variables));
                device_variables = nullptr;
            }
            if(cpt_array_device != nullptr) {
                gpuErrcheck(cudaFree(cpt_array_device));
                cpt_array_device = nullptr;
            }
            if(parents_array_device != nullptr){
                gpuErrcheck(cudaFree(parents_array_device));
                parents_array_device = nullptr;
            }
        }

    };
}

#endif //BAYLIB_CUDA_GRAPH_revised_REVISED_CUH
