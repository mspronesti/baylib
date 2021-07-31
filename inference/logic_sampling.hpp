//
// Created by elle on 22/07/21.
//

#ifndef GPUTEST_LOGIC_SAMPLING_HPP
#define GPUTEST_LOGIC_SAMPLING_HPP

#define DEBUG_MONTECARLO 0
#define BOOST_COMPUTE_THREAD_SAFE

#include <vector>
#include <future>

#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "../network/bayesian_network.h"

namespace bn {
    namespace compute = boost::compute;
	using boost::compute::lambda::_1;
    using boost::compute::lambda::_2;
	
    /* aliases */
    /// graph
    using graph = boost::adjacency_list<boost::setS, boost::setS, boost::bidirectionalS>;
    using node = graph::vertex_descriptor;
    
    /// graph ranking
    using rank_t = std::map<node, int>;

	struct bcvec {
        compute::vector<int> vec;
        int cardinality;
        bcvec(int dim, const compute::context& context, int cardinality): cardinality(cardinality){
            vec = compute::vector<int>(dim, context);
        }
    };

    template <typename T>
    class logic_sampling {
		
		
    public:
        logic_sampling() {
            this->device = compute::system::default_device();
            this->context = compute::context(device);
            this->queue = compute::command_queue(context, device);
            this->rand_eng = std::make_unique<compute::default_random_engine>(queue);
        }

        logic_sampling(const compute::device &device, const bn::bayesian_network<T> &bn){
            to_boost_graph(bn);
            // TODO: add missing
        }
		
		explicit logic_sampling(const std::shared_ptr<bn::bayesian_network<T>> &bn,
                const compute::device &device = compute::system::default_device()
                        ) {
            this->device = device;
            this->context = compute::context(device);
            this->queue = compute::command_queue(context, device);
            this->rand_eng = std::make_unique<compute::default_random_engine>(queue);

			// init bn, dipende da come scriviamo la BN 
        }




    private:
        graph g;
        std::vector<node> nodes;
		compute::device device;
        compute::context context;
        compute::command_queue queue;
        std::unique_ptr<compute::default_random_engine> rand_eng;
		
		
        // private members
        std::vector<T> accumulate_cpt(std::vector<T> striped_cpt, int possible_states);
        std::shared_ptr<bn::bcvec> simulate_node(const std::vector<T>& striped_cpt,
                                                                    const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                                                    int dim = 10000,
                                                                    int possible_states = 2);
        std::pair<int, int> compute_result_binary(bcvec &res);
        std::vector<int> compute_result_general(bcvec &res);

        rank_t graph_rank();
        bool exists_edge(node v1, node v2, const graph &g);
        void to_boost_graph(const bn::bayesian_network<T> &bn);

    };


#if DEBUG_MONTECARLO
    template<typename S>
    void logic_sampling<T>::print_vec(compute::vector<S> &vec, const std::string& message="", int len=-1){
        if(len == -1)
            len = vec.size();
        std::vector<S> host_vec(len);
        compute::copy(vec.begin(), vec.begin() + len, host_vec.begin(), queue);
        std::cout << message << ' ';
        for(T el: host_vec)
            std::cout << el << ' ';
        std::cout << '\n';
    }
#endif
    template <typename T>
    std::vector<T> logic_sampling<T>::accumulate_cpt(std::vector<T> striped_cpt, int possible_states){
        for(int i = 0 ; i < striped_cpt.size() ; i += possible_states)
            for(int j = 1 ; j < possible_states - 1 ; j++)
                striped_cpt[i + j] += striped_cpt[i + j - 1];
        return striped_cpt;
    }


    /// Node Simulation with GPU parallelization
    /// \param striped_cpt CPT table in a contiguous format
    /// \param parents_result output of parent nodes, if simulating source leave empty
    /// \param dim number of samples to simulate, it must be consistent with parents simulation
    /// \param possible_states cardinality of the discrete variable to simulate (e.g. 2 if binary variable)
    /// \return shared_ptr to result of simulation, use with other simulations or condense results with compute_result
    template <typename T>
    std::shared_ptr<bn::bcvec> logic_sampling<T>::simulate_node(const std::vector<T>& striped_cpt,
                                         const std::vector<std::shared_ptr<bcvec>>& parents_result,
                                         int dim,
                                         int possible_states){


        std::vector<T> striped_cpt_accum = this->accumulate_cpt(striped_cpt, possible_states);
        std::shared_ptr<bcvec> result = std::make_shared<bcvec>(dim, context, possible_states);
        compute::vector<T> device_cpt(striped_cpt.size(), context);
        compute::vector<T> threshold_vec(dim, context);
        compute::vector<T> random_vec(dim, context);
        compute::uniform_real_distribution<T> distribution(0, 1);
        compute::vector<int> index_vec(dim, context);

        compute::copy(striped_cpt_accum.begin(), striped_cpt_accum.end(), device_cpt.begin(), queue);

        if(parents_result.empty()){
            compute::fill(index_vec.begin(), index_vec.end(), 0, queue);
        }
        else {
            int coeff = possible_states;
            for (int i = 0; i < parents_result.size(); i++) {
#if DEBUG_MONTECARLO
                print_vec(*parents_result[i], "PARENT", 10);
#endif
                if (i == 0)
                    compute::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(), index_vec.begin(),
                                       _1 * coeff, queue);
                else
                    compute::transform(parents_result[i]->vec.begin(), parents_result[i]->vec.end(), index_vec.begin(),
                                   index_vec.begin(), _1 * coeff + _2, queue);
                coeff *= parents_result[i]->cardinality;
            }
        }
        compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
#if DEBUG_MONTECARLO
        print_vec(index_vec, "INDEX", 10);
        print_vec(threshold_vec, "THRESH", 10);
#endif
        distribution.generate(random_vec.begin(), random_vec.end(), *rand_eng, queue);
        compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), result->vec.begin(), _1 > _2, queue);
        for(int i = 0; i + 2 < possible_states ; i++){
            compute::vector<int> temp(dim, context);
            compute::transform(index_vec.begin(), index_vec.end(), index_vec.begin(), _1 + 1, queue);
            compute::gather(index_vec.begin(), index_vec.end(), device_cpt.begin(), threshold_vec.begin(), queue);
            compute::transform(random_vec.begin(), random_vec.end(), threshold_vec.begin(), temp.begin(), _1 > _2, queue);
            compute::transform(temp.begin(), temp.end(), result->vec.begin(), result->vec.begin(), _1 + _2, queue);
        }

#if DEBUG_MONTECARLO
        print_vec(random_vec, "RANDOM", 10);
        print_vec(*result, "RESULT", 10);
#endif
        return result;
    }

    /// Accumulate simulation results for binary case
    /// \param res result from simulate_node
    /// \return pair<Occurrences of 0, Occurrences of 1>
    template <typename T>
    std::pair<int, int> logic_sampling<T>::compute_result_binary(bcvec &res){
        int sum = 0;
        compute::reduce(res.vec.begin(), res.vec.end(), &sum, queue);
        return std::pair<int, int>(res.vec.size() - sum, sum);
    }

    /// Accumulate simulation results for general case
    /// \param res result from simulate node
    /// \return vector for witch the i-th element is the number of occurrences of i
    template <typename T>
    std::vector<int> logic_sampling<T>::compute_result_general(bcvec &res){
        std::vector<int> acc_res(res.cardinality);
        for (int i = 0; i < res.cardinality; ++i) {
            acc_res[i] = compute::count(res.vec.begin(), res.vec.end(), i, queue);
        }
        return acc_res;
    }

    /**
    * Applies ranking function to the DAG g
    * @tparam T
    * @return map containing node-rank as key-value couple
    */
    template<typename T>
    rank_t logic_sampling<T>::graph_rank() {
        std::vector<node> roots;
        rank_t ranks{};

        // fill nodes map with 0s
        for(auto & v : nodes)
            ranks[v] = 0;

        // find roots
        graph::vertex_iterator v, vend;
        for (auto vd : boost::make_iterator_range(boost::vertices(g)))
            if(boost::in_degree(vd, g) == 0)
                roots.push_back(vd);

        if(roots.empty())
            throw std::logic_error("No root nodes found in graph.");

        while(!roots.empty()) {
            node curr_node = roots.back();
            roots.pop_back();

            for (auto vd : boost::make_iterator_range(boost::vertices(g))) {
                if (!exists_edge(curr_node, vd, g)) continue;

                if (ranks[curr_node] + 1 > ranks[vd]) {
                    ranks[vd] = ranks[curr_node] + 1;
                    roots.push_back(vd);
                }
            }
        }

        return ranks;
    }


    /**
    * determines whether nodes v1 and v2 are adjacent
    * @tparam T
    * @param v1 : source node
    * @param v2 : destination node
    * @param g  : BGL adjacency list
    * @return true if v1's adjacent v2, false otherwise
    */
    template<typename T>
    bool logic_sampling<T>::exists_edge(bn::node v1, bn::node v2, const bn::graph &g) {
        return boost::edge(v1, v2, g).second;
    }

    /**
    * Extracts needed info from custom graph to build internal
    * graph using BGL
    * @tparam T
    * @param bn : bayesian network
    */
    template <typename T>
    void logic_sampling<T>::to_boost_graph(const bn::bayesian_network<T> &bn){
        // TODO: to be implemented
    }


} // namespace bn


#endif //GPUTEST_LOGIC_SAMPLING_HPP
