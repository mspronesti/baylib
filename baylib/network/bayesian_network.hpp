#ifndef BAYLIB_BAYESIAN_NETWORK_HPP
#define BAYLIB_BAYESIAN_NETWORK_HPP

#include <baylib/graph/graph.hpp>
#include <baylib/network/random_variable.hpp>
#include <baylib/baylib_assert.h>
#include <baylib/network/bayesian_utils.hpp>

//! \file bayesian_network.hpp
//! \brief bayesian network implementation based on boost GPL

namespace bn {

    /**
     * This class models a Bayesian Network allowing both
     * index and name based access to its facilities for a better
     * user experience
     * @tparam Probability : Type of cpts entries
     */
    template <typename Probability>
    class bayesian_network {
        typedef bn::graph<bn::random_variable<Probability>> graph_t;
        typedef bn::vertex<Probability> vertex_id;

    public:
        bayesian_network() : graph(std::make_shared<graph_t>()) { }

        // overloading begin and end to easily loop over random_variables
        // avoiding packing copies inside other facilities
        auto begin(){
            return bn::bundles(*graph).begin();
        }

        auto begin() const {
            return bn::bundles(*graph).begin();
        }

        auto end() {
            return bn::bundles(*graph).end();
        }

        auto end() const {
            return bn::bundles(*graph).end();
        }

        /**
         * Method to add a new variable to the network, no duplicate names can exist in the graph
         * @param name   : string identifier of the variable
         * @param states : vector of possible states that can be obtained from the random distribution
         * @return       : numerical identifier assigned to the node
         */
        vertex_id add_variable(const std::string &name, const std::vector<std::string> &states){
            BAYLIB_ASSERT(!has_variable(name),
                          "random_variable with name "
                          + name + " already exists",
                          std::runtime_error )

            vertex_id v = boost::add_vertex(bn::random_variable<Probability>{name, states}, *graph);
            var_map[name] = v;
            variable(name)._id = v;
            return v;
        }

        /**
         * Method to remove a variable from the network, dependencies are also automatically deleted
         * @param name name of variable
         */
        void remove_variable(const std::string &name){
            auto v = index_of(name);
            boost::remove_vertex(v, *graph);

            for (auto & var : *this)
                if (has_dependency(name, var.name()))
                    var.parents_info.remove(name);

            var_map.erase(name);
        }

        /**
         * Method to add a dependency / arc between two nodes of the network
         * @param src_name  : parent node name
         * @param dest_name : child node name
         */
        void add_dependency(const std::string &src_name, const std::string &dest_name){
            auto src_id = index_of(src_name);
            auto dest_id = index_of(dest_name);

            add_dependency(src_id, dest_id);
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

            auto& src = variable(src_id);
            auto& dest = variable(dest_id);

            boost::add_edge(src_id, dest_id, *graph);
            dest.parents_info.add(src._name, src.states().size());
        }

        /**
         * Method to remove a dependency / arc between two nodes of the network
         * @param src_name  : parent node name
         * @param dest_name : child node name
         */
        void remove_dependency(const std::string &src_name, const std::string &dest_name){
            auto src_id = index_of(src_name);
            auto dest_id = index_of(dest_name);

            remove_dependency(src_id, dest_id);
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

            auto src_name = variable(src_id)._name;
            auto &dest = variable(dest_id);

            boost::remove_edge(src_id, dest_id, *graph);
            dest.parents_info.remove(src_name);
        }

        /**
         * Method that returns the number of variables in the network
         * @return number of variables
         */
        std::uint64_t number_of_variables() const {
            return boost::num_vertices(*graph);
        }

        /**
         * Operator that returns the random_variable class identified by the name specified
         * @param name : name string
         * @return     : random variable class
         */
        bn::random_variable<Probability>& operator [] (const std::string &name){
            auto v  = index_of(name);
            return (*graph)[v];
        }

        /**
         * Operator that returns the random_variable class identified by the name specified
         * @param name : name string
         * @return     : random variable class
         */
        bn::random_variable<Probability>& operator [] (const std::string &name) const{
            auto v  = index_of(name);
            return (*graph)[v];
        }

        /**
         * Operator that returns the random_variable class identified by id
         * @param name : name string
         * @return     : random variable class
         */
        bn::random_variable<Probability>& operator [] (vertex_id v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        /**
         * Operator that returns the random_variable class identified by id
         * @param name : name string
         * @return     : random variable class
         */
        bn::random_variable<Probability>& operator [] (vertex_id v) const{
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        /**
         * Method that returns the random_variable class identified by the name specified
         * @param name : name string
         * @return     : random variable class
         */
        bn::random_variable<Probability> & variable(const std::string &name){
            auto v  = index_of(name);
            return (*graph)[v];
        }

        /**
         * Operator that returns the random_variable class identified by id
         * @param name : name string
         * @return     : random variable class
         */
        bn::random_variable<Probability> & variable(vertex_id v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        /**
         * Method to verify if two variables, identified by name, are connected by a dependency / arc
         * @param name1 : name of first node
         * @param name2 : name of second node
         * @return      : true if v1 is parent of v2
         */
        bool has_dependency(const std::string &name1, const std::string &name2)  {
            auto v1  = index_of(name1);
            auto v2  = index_of(name2);

            return boost::edge(v1, v2, *graph).second;
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
         * Method to verify if a node, identified by name, is a root node / has no parent nodes
         * @param name : name string
         * @return     : true if node is root
         */
        bool is_root(const std::string &name) const {
            auto v  = index_of(name);
            return boost::in_degree(v, *graph) == 0
                   && boost::out_degree(v, *graph) != 0;
        }

        /**
         * Return the vector of children of a specific node identified by name
         * @param name : name string
         * @return     : vector of numerical ids
         */
        std::vector<vertex_id> children_of(const std::string &name) const{
            auto v  = index_of(name);
            return children_of(v);
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
         * Return the vector of parents of a specific node identified by name
         * @param name : name of node
         * @return     : vector of numerical ids
         */
        std::vector<vertex_id> parents_of(const std::string &name) const{
            auto v = index_of(name);
            return parents_of(v);
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
         * Sets the probability value of a CPT cell of a specific node, identified by name
         * @param var_name    : name of node
         * @param state_value : identifier of state associated with the set probability
         * @param cond        : condition associated with the set probability
         * @param p           : probability
         */
        void set_variable_probability(
                const std::string& var_name,
                bn::state_t state_value,
                const bn::condition& cond,
                Probability p
        )
        {
            auto & var = variable(var_name);
            auto nparents = parents_of(var._id).size();

            // make sure the cardinality of parents is correct
            BAYLIB_ASSERT(cond.size() == nparents,
                          "condition contains "
                          << cond.size() << " while "
                          << var_name << " has "
                          << nparents,
                          std::logic_error)

            // make sure the parents are actually correct
            BAYLIB_ASSERT(std::all_of(cond.begin(), cond.end(),
                                      [this, var_name](const auto&c){
                                          return has_dependency(c.first, var_name);
                                      }),
                          "no such parent for variable "
                          + var_name,
                          std::runtime_error)

            var.set_probability(state_value, cond, p);

            if(cpt_filled_out(var))
                optimize_cpt_memory_occupation(var._id);
        }

        /**
         * Method to verify if a name is associated with an existing variable, identified by name
         * @param name : name of variable
         * @return     : true if variable exists
         */
        bool has_variable(const std::string &name) const {
            return var_map.find(name) != var_map.end();
        }

        /**
         * Method to verify if a name is associated with an existing variable, identified by numerical identifier
         * @param v : numerical identifier of variable
         * @return  : true if variable exists
         */
        bool has_variable(vertex_id v) const {
            return boost::num_vertices(*graph) > v;
        }


        /**
         * Returns the numerical identifier of a variable identified by name
         * @param name : name of variable
         * @return     : numerical identifier of variable
         */
        vertex_id index_of(const std::string &name) const {
            auto it = var_map.find(name);
            BAYLIB_ASSERT(it != var_map.end(),
                          "identifier " + name + " doesn't "
                          "represent a random_variable",
                          std::runtime_error)

            return it->second;
        }

    private:
        std::shared_ptr<graph_t> graph;
        std::map<std::string, vertex_id> var_map;
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
} // namespace bn

#endif //BAYLIB_BAYESIAN_NETWORK_HPP