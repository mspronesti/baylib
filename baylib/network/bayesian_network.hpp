#ifndef BAYLIB_BAYESIAN_NETWORK_HPP
#define BAYLIB_BAYESIAN_NETWORK_HPP

#include <baylib/graph/graph.hpp>
#include <baylib/network/random_variable.hpp>
#include <baylib/baylib_assert.h>
#include <baylib/network/bayesian_utils.hpp>

/**
 * ================ Bayesian Network ===================
 * This class represents a Bayesian Network allowing both
 * index and name based access to its facilities for a better
 * user experience
 */

namespace bn {

    template <typename Probability>
    class bayesian_network {
        typedef bn::graph<bn::random_variable<Probability>> graph_t;
        typedef bn::vertex<Probability> vertex_id;

    public:
        bayesian_network() : graph(std::make_shared<graph_t>()){}

        // constructor from xdls file
        explicit bayesian_network(const std::string & xdls_filename) {
            // TODO: to be implemented
        }


        ~bayesian_network() {
            graph.reset();
        }

        bool operator == (const bayesian_network &bn) const {
            return graph.get() == bn.graph.get();
        }

        void add_variable(const std::string &name, const std::vector<std::string> &states){
            BAYLIB_ASSERT(!has_variable(name),
                          "random_variable with name "
                          + name + " already exists",
                          std::runtime_error )

            vertex_id v = boost::add_vertex(bn::random_variable<Probability>{name, states}, *graph);
            var_map[name] = v;
            variable(name)._id = v;
        }

        void remove_variable(const std::string &name){
            auto v = index_of(name);
            boost::remove_vertex(v, *graph);

            var_map.erase(name);
            for (auto & var : variables())
                if(has_dependency(name, var.name()))
                    var.parents_info.remove(name);
        }

        void add_dependency(const std::string &src_name, const std::string &dest_name){
            auto& src = variable(src_name);
            auto& dest = variable(dest_name);

            BAYLIB_ASSERT(!introduces_loop(dest._id, src._id),
                          "adding conditional dependency "
                          + src_name + " to " + dest_name +
                          " would introduce a loop",
                          std::logic_error )

            boost::add_edge(src._id, dest._id, *graph);
            dest.parents_info.add(src_name, src.states().size());
        }

        void add_dependency(vertex_id src_id, vertex_id dest_id){
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

        void remove_dependency(const std::string &src_name, const std::string &dest_name){
            BAYLIB_ASSERT(has_variable(src_name) && has_variable(dest_name),
                          "out of bound access to graph",
                          std::out_of_range)

            auto& src = variable(src_name);
            auto& dest = variable(dest_name);

            boost::remove_edge(src._id, dest._id, *graph);
            dest.parents_info.remove(src_name);
        }

        void remove_dependency(vertex_id src_id, vertex_id dest_id) {
            BAYLIB_ASSERT(has_variable(src_id) && has_variable(dest_id),
                          "out of bound access to graph",
                          std::out_of_range)

            auto src_name = variable(src_id)._name;
            auto &dest = variable(dest_id);

            boost::remove_edge(src_id, dest_id, *graph);
            dest.parents_info.remove(src_name);
        }

        std::vector<bn::random_variable<Probability>> variables() const {
            auto vars = std::vector<bn::random_variable<Probability>>{};
            for(auto v : boost::make_iterator_range(boost::vertices(*graph)))
                vars.push_back((*graph)[v]);

            return vars;
        }

        std::uint64_t number_of_variables() const {
            return boost::num_vertices(*graph);
        }

        bn::random_variable<Probability>& operator [] (const std::string &name){
            auto v  = index_of(name);
            return (*graph)[v];
        }

        bn::random_variable<Probability>& operator [] (const std::string &name) const{
            auto v  = index_of(name);
            return (*graph)[v];
        }

        bn::random_variable<Probability>& operator [] (vertex_id v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        bn::random_variable<Probability>& operator [] (vertex_id v) const{
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        bn::random_variable<Probability> & variable(const std::string &name){
            auto v  = index_of(name);
            return (*graph)[v];
        }

        bn::random_variable<Probability> & variable(vertex_id v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        bool has_dependency(const std::string &name1, const std::string &name2)  {
            auto v1  = index_of(name1);
            auto v2  = index_of(name2);

            return boost::edge(v1, v2, *graph).second;
        }

        bool has_dependency(vertex_id v1, vertex_id v2) const {
            BAYLIB_ASSERT(has_variable(v1) && has_variable(v2),
                          "out of bound access to graph",
                          std::out_of_range)

            return boost::edge(v1, v2, *graph).second;
        }

        bool is_root(vertex_id v) const {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return boost::in_degree(v, *graph) == 0
                  && boost::out_degree(v, *graph) != 0;
        }

        bool is_root(const std::string &name) const {
            auto v  = index_of(name);
            return boost::in_degree(v, *graph) == 0
                   && boost::out_degree(v, *graph) != 0;
        }

        std::vector<vertex_id> children_of(const std::string &name) const{
            auto v  = index_of(name);
            auto it = boost::make_iterator_range(adjacent_vertices(v, *graph));

            return std::vector<vertex_id>(it.begin(), it.end());
        }

        std::vector<vertex_id> children_of(vertex_id v) const{
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            auto it = boost::make_iterator_range(adjacent_vertices(v, *graph));
            return std::vector<vertex_id>(it.begin(), it.end());
        }

        std::vector<vertex_id> parents_of(const std::string &name) const{
            auto v = index_of(name);
            std::vector<vertex_id> parents;

            for(auto ed : boost::make_iterator_range(boost::in_edges(v, *graph)))
                parents.push_back(boost::source(ed, *graph));

            return parents;
        }


        std::vector<vertex_id> parents_of(vertex_id v) const{
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            std::vector<vertex_id> parents;

            for(auto ed : boost::make_iterator_range(boost::in_edges(v, *graph)))
                parents.push_back(boost::source(ed, *graph));

            return parents;
        }

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
                          + std::to_string(cond.size())
                          + " while " + var_name + " has "
                          + std::to_string(nparents),
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

        bool has_variable(const std::string &name) const {
            return var_map.find(name) != var_map.end();
        }

        bool has_variable(vertex_id v) const {
            return boost::num_vertices(*graph) > v;
        }

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