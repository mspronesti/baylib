#ifndef BAYLIB_BAYESIAN_NETWORK_HPP
#define BAYLIB_BAYESIAN_NETWORK_HPP

#include <baylib/graph/graph.hpp>
#include <baylib/network/random_variable.hpp>
#include <baylib/assert.h>

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
    public:
        bayesian_network() : graph(std::make_shared<graph_t>()){}

        // constructor from xdls file
        explicit bayesian_network(const std::string & xdls_filename) {
            // TODO: to be implemented
        }

        /*
        bayesian_network(const bayesian_network &bn){
            graph = bn.graph;
        }
        */

        ~bayesian_network() {
            graph.reset();
        }
        /*
        bayesian_network & operator = (const bayesian_network &bn){
            if(this != &bn){
                graph = bn.graph;
            }
            return *this;
        }
        */

        bool operator == (const bayesian_network &bn) const {
            return graph.get() == bn.graph.get();
        }

        void add_variable(const std::string &name, const std::vector<std::string> &states){
            BAYLIB_ASSERT(!has_variable(name),
                          "random_variable with name "
                          + name + " already exists",
                          std::runtime_error )

            bn::vertex<Probability> v = boost::add_vertex(bn::random_variable<Probability>{name, states}, *graph);
            var_map[name] = v;
            variable(name)._id = v;
        }

        void remove_variable(const std::string &name){
            auto v = index_of(name);
            boost::remove_vertex(v, *graph);
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
            dest.parents_states[src_name] = src.states().size();
        }

        void add_dependency(const bn::vertex<Probability> &src_id, const bn::vertex<Probability> &dest_id){
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
            dest.parents_states[src._name] = src.states().size();
        }

        void remove_dependency(const bn::vertex<Probability> &src_id, const bn::vertex<Probability> &dest_id) {
            BAYLIB_ASSERT(has_variable(src_id) && has_variable(dest_id),
                          "out of bound access to graph",
                          std::out_of_range)

            auto src_name = variable(src_id)._name;
            auto &dest = variable(dest_id);

            boost::remove_edge(src_id, dest_id);
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

        bn::random_variable<Probability>& operator [] (bn::vertex<Probability> v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        bn::random_variable<Probability>& operator [] (bn::vertex<Probability> v) const{
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return (*graph)[v];
        }

        bn::random_variable<Probability> & variable(const std::string &name){
            auto v  = index_of(name);
            return (*graph)[v];
        }

        bn::random_variable<Probability> & variable(bn::vertex<Probability> v) {
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

        bool has_dependency(bn::vertex<Probability> v1, bn::vertex<Probability> v2) const {
            BAYLIB_ASSERT(has_variable(v1) && has_variable(v2),
                          "out of bound access to graph",
                          std::out_of_range)

            return boost::edge(v1, v2, *graph).second;
        }

        bool is_root(bn::vertex<Probability> v) const {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            return boost::in_degree(v, *graph) == 0;
        }

        bool is_root(const std::string &name) const {
            auto v  = index_of(name);
            return boost::in_degree(v, *graph) == 0;
        }

        std::vector<bn::vertex<Probability>> children_of(const std::string &name) {
            auto v  = index_of(name);
            auto it = boost::make_iterator_range(adjacent_vertices(v, *graph));

            return std::vector<bn::vertex<Probability>>(it.begin(), it.end());
        }

        std::vector<bn::vertex<Probability>> children_of(bn::vertex<Probability> v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            auto it = boost::make_iterator_range(adjacent_vertices(v, *graph));
            return std::vector<bn::vertex<Probability>>(it.begin(), it.end());
        }

        std::vector<bn::vertex<Probability>> parents_of(const std::string &name) {
            auto v = index_of(name);
            std::vector<bn::vertex<Probability>> parents;

            for(auto ed : boost::make_iterator_range(boost::in_edges(v, *graph)))
                parents.push_back(boost::source(ed, *graph));

            return parents;
        }


        std::vector<bn::vertex<Probability>> parents_of(bn::vertex<Probability> v) {
            BAYLIB_ASSERT(has_variable(v),
                          "out of bound access to graph",
                          std::out_of_range)

            std::vector<bn::vertex<Probability>> parents;

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
            [[maybe_unused]] auto var = index_of(var_name); // to assert var existence
            auto nparents = parents_of(var).size();

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

            this->variable(var_name).set_probability(state_value, cond, p);
        }

        bool has_variable(const std::string &name) const {
            return var_map.find(name) != var_map.end();
        }

        bool has_variable(bn::vertex<Probability> v) const {
            return boost::num_vertices(*graph) > v;
        }

        bn::vertex<Probability> index_of(const std::string &name) const {
            auto it = var_map.find(name);
            BAYLIB_ASSERT(it != var_map.end(),
                          "identifier " + name + " doesn't "
                          "represent a random_variable",
                          std::runtime_error)

            return it->second; // more efficient than calling "at"
        }


    private:
        std::shared_ptr<graph_t> graph;
        std::map<std::string, vertex<Probability>> var_map;

        /**
         * utility to detect whether a vertex introduces
         * a loop in the DAG
         * @param from : edge's driver
         * @param to  : edge's loader
         * @return true if introduces a loop, false otherwise
         */
        bool introduces_loop(const bn::vertex<Probability> &from, const bn::vertex<Probability> &to){
            if(from == to) return true;

            auto const children = children_of(from);
            return std::any_of(
                    children.cbegin(), children.cend(),
                    [this, &to](const bn::vertex<Probability> &next){
                        return introduces_loop(next, to);
                    });
        }
    };
} // namespace bn

#endif //BAYLIB_BAYESIAN_NETWORK_HPP