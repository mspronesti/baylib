#ifndef BAYLIB_BAYESIAN_NET_HPP
#define BAYLIB_BAYESIAN_NET_HPP

#include <baylib/graph/graph.hpp>
#include <baylib/network/random_variable.hpp>
#include <baylib/baylib_assert.h>
#include <baylib/network/bayesian_utils.hpp>
#include <baylib/baylib_concepts.hpp>

//! \file bayesian_net.hpp
//! \brief bayesian network implementation based on boost GPL

namespace baylib {
    /**
     * This class models a Bayesian Network allowing both
     * index and name based access to its facilities for a better
     * user experience
     * @tparam Probability : Type of cpts entries
     */
    template <RVarDerived Variable_>
    class bayesian_net {
        typedef baylib::graph<Variable_> graph_type;
        typedef baylib::vertex<Variable_> vertex_id;
    public:
        typedef Variable_ variable_type;
        typedef typename variable_type::probability_type probability_type;

        bayesian_net() : graph(std::make_shared<graph_type>()) { }

        // overloading begin and end to easily loop over random_variables
        // avoiding packing copies inside other facilities
        auto begin(){
            return baylib::bundles(*graph).begin();
        }

        auto begin() const {
            return baylib::bundles(*graph).begin();
        }

        auto end() {
            return baylib::bundles(*graph).end();
        }

        auto end() const {
            return baylib::bundles(*graph).end();
        }

        /**
         * adds a new variable to the bayesian network
         * @tparam A
         * @param args   : vararg to specify the Variable constructor parameters
         *                 according to the template parameter
         * @return       : numerical identifier assigned to the node
         */
        template <typename ...Args_>
#ifdef __concepts_supported
        requires std::is_constructible_v<Variable_, Args_&&...>
#endif
        vertex_id add_variable(Args_ &&...args) {
#ifndef __concepts_supported
            static_assert(std::is_constructible_v<Variable_, Args_&&...>);
#endif
            Variable_ var(std::forward<Args_>(args)...);
            vertex_id v = boost::add_vertex(std::move(var), *graph);
            (*graph)[v]._id = v;
            return v;
        }

        /**
         * Method to remove a variable from the network, dependencies are also automatically
         * deleted
         * @param var_id: id of variable
         */
        void remove_variable(vertex_id var_id) {
            boost::remove_vertex(var_id, *graph);
        }


        /**
         * Method to add a dependency / arc between two nodes of the network
         * @param src_id  : parent node identifier
         * @param dest_id : child node identifier
         */
        void add_dependency(vertex_id src_id, vertex_id dest_id) {
            BAYLIB_ASSERT(has_variable(src_id) && has_variable(dest_id),
                          "out of bound access to vertices",
                          std::out_of_range);

            BAYLIB_ASSERT(!introduces_loop(dest_id, src_id),
                          "adding conditional dependency "
                          " would introduce a loop",
                          std::logic_error )

            boost::add_edge(src_id, dest_id, *graph);
        }


        /**
         * Method to remove a dependency / arc between two nodes of the network
         * @param src_id  : parent node identifier
         * @param dest_id : child node identifier
         */
        void remove_dependency(vertex_id src_id, vertex_id dest_id) {
            BAYLIB_ASSERT(has_variable(src_id) && has_variable(dest_id),
                          "out of bound access to graph",
                          std::out_of_range)

            boost::remove_edge(src_id, dest_id, *graph);
        }

        /**
         * Method that returns the number of variables in the network
         * @return number of variables
         */
        unsigned long number_of_variables() const {
            return boost::num_vertices(*graph);
        }


        /**
         * Operator that returns the random_variable class identified by id
         * @param v : variable id
         * @return  : random variable class
         */
        variable_type & operator [] (vertex_id v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        /**
         * Operator that returns the random_variable class identified by id
         * @param  v: variable id
         * @return  : random variable class
         */
        variable_type & operator [] (vertex_id v) const {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        /**
         * Operator that returns the random_variable class identified by id
         * @param v : variable id
         * @return  : random variable class
         */
        variable_type & variable(vertex_id v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }


        /**
         * Method to verify if two variables, identified by numerical id, are connected by a dependency / arc
         * @param v1 : id of first node
         * @param v2 : id of second node
         * @return   : true if v1 is parent of v2
         */
        bool has_dependency(vertex_id v1, vertex_id v2) const {
            BAYLIB_ASSERT(has_variable(v1) && has_variable(v2),
                          "out of bound access to graph",
                          std::out_of_range)

            return boost::edge(v1, v2, *graph).second;
        }

        /**
         * Method to verify if a node, identified by numerical identifier, is a root node / has no parent nodes
         * @param v : id of node
         * @return  : true if node is root
         */
        bool is_root(vertex_id v) const {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return boost::in_degree(v, *graph) == 0
                  && boost::out_degree(v, *graph) != 0;
        }

        /**
         * Return the vector of children of a specific node identified by numerical identifier
         * @param v : id of node
         * @return  : vector of numerical ids
         */
        std::vector<vertex_id> children_of(vertex_id v) const{
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            auto it = boost::make_iterator_range(adjacent_vertices(v, *graph));
            return std::vector<vertex_id>(it.begin(), it.end());
        }

        /**
         * Return the vector of parents of a specific node identified by numerical id
         * @param v : numerical id of node
         * @return  : vector of numerical ids
         */
        std::vector<vertex_id> parents_of(vertex_id v) const{
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            std::vector<vertex_id> parents;

            for(auto ed : boost::make_iterator_range(boost::in_edges(v, *graph)))
                parents.push_back(boost::source(ed, *graph));

            return parents;
        }

        /**
         * Sets the probability value of a CPT cell of a specific node, identified by id
         * @param var_id      : id of node
         * @param state_value : identifier of state associated with the set probability
         * @param cond        : condition associated with the set probability
         * @param p           : probability
         */
        void set_variable_probability(
                const vertex_id var_id,
                baylib::state_t state_value,
                const baylib::condition& cond,
                probability_type p
        )
        {
            auto nparents = parents_of(var_id).size();

            // make sure the cardinality of parents is correct
            BAYLIB_ASSERT(cond.size() == nparents,
                          "condition contains "
                          << cond.size() << " while "
                          "variable " << var_id
                          << " has " << nparents,
                          std::logic_error)

            // make sure the parents are actually correct
            BAYLIB_ASSERT(std::all_of(cond.begin(), cond.end(),
                                      [this, var_id](const auto&c){
                                          return has_dependency(c.first, var_id);
                                      }),
                          "no such parent for variable "
                          << var_id,
                          std::runtime_error)

            (*graph)[var_id].set_probability(state_value, cond, p);

            if(cpt_filled_out(*this, var_id))
                optimize_cpt_memory_occupation(var_id);
        }

        /**
         * Method to verify if a numeric identifier
         * is associated with an existing variable
         * @param v : numerical identifier of variable
         * @return  : true if variable exists
         */
        bool has_variable(vertex_id v) const {
            return boost::num_vertices(*graph) > v;
        }


    private:
        std::shared_ptr<graph_type> graph;
        std::unordered_map<size_t, vertex_id> cpt_hash_map;

        /**
         * utility to detect whether a vertex introduces
         * a loop in the DAG
         * @param from : edge's driver
         * @param to  : edge's loader
         * @return true if introduces a loop, false otherwise
         */
        bool introduces_loop(vertex_id from, vertex_id to){
            if(from == to) return true;

            auto const children = children_of(from);
            return std::any_of(
                    children.cbegin(), children.cend(),
                    [this, &to](vertex_id next){
                        return introduces_loop(next, to);
                    });
        }


        /**
         * Checks if a specific variable, identified by numerical identifier, has a cpt that is already
         * present in the network, if it's the case the cpt of the node is deleted from memory and switched
         * with a reference to the one that already exists, the cpt will be copied if a write operation is executed
         * following the COW paradigm
         * @param id : numerical identifier of node
         */
        void optimize_cpt_memory_occupation(vertex_id id){
            auto seed = variable(id).cpt.hash();
            if(cpt_hash_map.find(seed) != cpt_hash_map.end()){
                auto var = variable(cpt_hash_map[seed]);
                if(var._id != id && var.cpt == variable(id).cpt){
                    variable(id).cpt.d = var.cpt.d;
                    return;
                }
            }
            cpt_hash_map[seed] = id;
        }
    };
} // namespace baylib

#endif //BAYLIB_BAYESIAN_NET_HPP