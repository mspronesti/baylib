#ifndef BAYESIAN_INFERRER_BAYESIAN_NETWORK_HPP
#define BAYESIAN_INFERRER_BAYESIAN_NETWORK_HPP

#include <baylib/graph/graph.hpp>
#include <baylib/probability/cpt.hpp>
#include <baylib/assert.h>

namespace bn {

    template <typename Probability>
    class bayesian_network {
    public:
        bayesian_network() : graph(std::make_shared<bn::graph<Probability>>()){}

        // constructor from xdls file
        explicit bayesian_network(const std::string & xdls_filename) {
            // TODO: to be implemented
        }

        bayesian_network(const bayesian_network &bn){
            graph = bn.graph;
        }

        ~bayesian_network() {
            graph.reset();
        }

        bayesian_network & operator = (const bayesian_network &bn){
            if(this != &bn){
                graph = bn.graph;
            }
            return *this;
        }

        bool operator == (const bayesian_network &bn) const {
            return graph.get() == bn.graph.get();
        }

        void add_variable(const std::string &name, const std::vector<std::string> &states){
            BAYLIB_ASSERT(!is_variable(name),
                          "random_variable with name "
                          + name + " already exists",
                          std::runtime_error )

            bn::vertex<Probability> v = boost::add_vertex(bn::random_variable<Probability>{name, states}, *graph);
            (*graph)[v].id = v;
            var_map[name] = v;
        }

        void remove_variable(const std::string &name){
            bn::vertex<Probability> v = find_variable(name);
            boost::remove_vertex(v, *graph);
        }

        void add_dependency(const std::string &name1, const std::string &name2){
            bn::vertex<Probability> from = var_map.at(name1);
            bn::vertex<Probability> to = var_map.at(name2);

            BAYLIB_ASSERT(!introduces_loop(to, from),
                          "adding conditional dependency "
                          + name1 + " to " + name2 +
                          " would introduce a loop",
                          std::logic_error )

            boost::add_edge(from, to, *graph);
        }

        void add_dependency(const bn::vertex<Probability> &v1, const bn::vertex<Probability> &v2){
            BAYLIB_ASSERT(!introduces_loop(v2, v1),
                          "adding conditional dependency "
                          " would introduce a loop",
                          std::logic_error )

            boost::add_edge(v1, v2, *graph);
        }

        void remove_dependency(const bn::vertex<Probability> &v1, const bn::vertex<Probability> &v2){
            boost::remove_edge(v1, v2);
        }

        std::vector<bn::random_variable<Probability>> variables() const {
            auto vars = std::vector<bn::random_variable<Probability>>{};

            for(auto v : boost::make_iterator_range(boost::vertices(*graph)))
                vars.push_back((*graph)[v]);

            return vars;
        }

        bn::random_variable<Probability>& operator [] (const std::string &name){
            auto v  = find_variable(name);
            return (*graph)[v];
        }

        bn::random_variable<Probability>& operator [] (const std::string &name) const{
            auto v  = find_variable(name);
            return (*graph)[v];
        }

        bool has_dependency(const std::string &name1, const std::string &name2)  {
            auto v1  = find_variable(name1);
            auto v2  = find_variable(name2);

            return boost::edge(v1, v2, *graph).second;
        }

        bool has_dependency(const bn::vertex<Probability> &v1, const bn::vertex<Probability> &v2) const {
            return boost::edge(v1, v2, *graph).second;
        }

        bool is_root(const bn::vertex<Probability> &v){
            return boost::in_degree(v, *graph) == 0;
        }

        bool is_root(const std::string &name){
            auto v  = find_variable(name);
            return boost::in_degree(v, *graph) == 0;
        }

        std::vector<bn::vertex<Probability>> children_of(const std::string &name) {
            auto v  = find_variable(name);
            std::vector<bn::vertex<Probability>> children{};
            for (auto vd : boost::make_iterator_range(adjacent_vertices(v, *graph)))
                children.push_back(vd);

            return children;
        }

        std::vector<bn::vertex<Probability>> children_of(const bn::vertex<Probability> &v) {
            std::vector<bn::vertex<Probability>> children{};
            for (auto vd : boost::make_iterator_range(adjacent_vertices(v, *graph)))
                children.push_back(vd);

            return children;
        }

        std::vector<bn::vertex<Probability>> parents_of(const std::string &name) {
            auto v = find_variable(name);
            std::vector<bn::vertex<Probability>> parents;

            for(auto vd : boost::make_iterator_range(boost::in_edges(v, *graph)))
                parents.push_back(boost::source(vd, *graph));

            return parents;
        }


        std::vector<bn::vertex<Probability>> parents_of(const bn::vertex<Probability> &v) {
            std::vector<bn::vertex<Probability>> parents;

            for(auto vd : boost::make_iterator_range(boost::in_edges(v, *graph)))
                parents.push_back(boost::source(vd, *graph));

            return parents;
        }


        bn::vertex<Probability> index_of(const std::string &name){
            return find_variable(name);
        }

        void set_variable_probability(
           const std::string& var_name,
           bn::state_t state_value,
           const bn::condition& cond,
           Probability p
        )
        {
            [[maybe_unused]] auto var = find_variable(var_name); // to assert var existence
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

            this->operator[](var_name).set_probability(state_value, cond, p);
        }

    private:
        std::shared_ptr<bn::graph<Probability>> graph;
        std::map<std::string, vertex<Probability>> var_map;

        bool is_variable(const std::string &name){
            return var_map.find(name) != var_map.end();
        }

        bn::vertex<Probability> find_variable(const std::string &name){
            auto it = var_map.find(name);
            BAYLIB_ASSERT(it != var_map.end(),
                          "identifier " + name + " doesn't "
                          "represent a random_variable",
                          std::runtime_error)

            return it->second; // more efficient than calling "at"
        }

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

#endif //BAYESIAN_INFERRER_BAYESIAN_NETWORK_HPP