#ifndef BAYESIAN_INFERRER_BAYESIAN_NETWORK_HPP
#define BAYESIAN_INFERRER_BAYESIAN_NETWORK_HPP
#include <baylib/graph/graph.hpp>
#include <baylib/network/probability/cpt.hpp>
namespace bn {

    template <typename Probability>
    class bayesian_network {
    public:
        bayesian_network() : graph(std::make_shared<bn::graph<Probability>>()){}

        bayesian_network(const bayesian_network &bn){
            graph = bn.graph;
        }

        ~bayesian_network() {
            graph.reset(); // credo evitabile !
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

        void add_variable(const std::string &name, const std::size_t nstates = 2){
            if(is_variable(name))
                throw std::runtime_error("variable with name " + name + " already exists");

            bn::vertex<Probability> v = boost::add_vertex(*graph);
            (*graph)[v].name = name;
            (*graph)[v].nstates = nstates;
            (*graph)[v].id = v;
            var_map[name] = std::move(v);
        }


        void remove_variable(const std::string &name){
            bn::vertex<Probability> v = find_variable(name);
            boost::remove_vertex(v, *graph);
        }

        void add_dependency(const std::string &name1, const std::string &name2){
            bn::vertex<Probability> from = var_map.at(name1);
            bn::vertex<Probability> to = var_map.at(name2);

            if(introduces_loop(to, from))
                throw std::logic_error("can't create a loop in DAG");

            boost::add_edge(from, to, *graph);
        }

        void add_dependency(const bn::vertex<Probability> &from, const bn::vertex<Probability> &to){
            if(introduces_loop(from, to))
                throw std::logic_error("can't create a loop in DAG");

            boost::add_edge(from, to, *graph);
        }

        void remove_dependency(const bn::vertex<Probability> &v1, const bn::vertex<Probability> &v2){
            boost::remove_edge(v1, v2);
        }

        bool conditional_dependency(const std::string &name1, const std::string &name2)  {
            auto v1  = find_variable(name1);
            auto v2  = find_variable(name2);

            return boost::edge(v1, v2, *graph).second;
        }

        bool conditional_dependency(const bn::vertex<Probability> &v1, const bn::vertex<Probability> &v2) const {
            return boost::edge(v1, v2, *graph).second;
        }

        bn::cpt<Probability> cpt_of(const std::string &name) const {
            if(!is_variable(name))
                throw std::runtime_error("identifier " + name + "doesn't represent a variable");
            return cpt_map[name];
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
            return children_of(v);
        }

        std::vector<bn::vertex<Probability>> children_of(const bn::vertex<Probability> &v) {
            std::vector<bn::vertex<Probability>> children{};
            for (auto vd : boost::make_iterator_range(adjacent_vertices(v, *graph)))
                children.push_back(vd);

            return children;
        }

        std::vector<bn::vertex<Probability>> parents_of(const std::string &name) {
            auto v  = find_variable(name);
            return parents_of(v);
        }

        std::vector<bn::vertex<Probability>> parents_of(const bn::vertex<Probability> &v) {
            std::vector<bn::vertex<Probability>> parents;

            for(auto vd : boost::make_iterator_range(boost::in_edges(v, *graph)))
                parents.push_back(boost::source(vd, *graph));

            return parents;
        }


        std::vector<bn::variable<Probability>> variables() const {
            auto vars = std::vector<bn::variable<Probability>>{};

            for(auto v : boost::make_iterator_range(boost::vertices(*graph)))
                vars.push_back((*graph)[v]);

            return vars;
        }

        bn::variable<Probability> getVariable(const std::string &name) {
            auto v  = find_variable(name);
            return (*graph)[v];
        }


    private:
        std::unique_ptr<bn::graph<Probability>> graph;
        std::map<std::string, vertex<Probability>> var_map;
        std::map<std::string, bn::cpt<Probability>> cpt_map;

        bool is_variable(const std::string &name){
            return var_map.find(name) != var_map.end();
        }

        bn::vertex<Probability> find_variable(const std::string &name){
            if(!is_variable(name))
                throw std::runtime_error("identifier " + name + " doesn't represent a variable");

            return var_map.at(name);
        }

        /**
         * utility to detect whether a vertex introduces
         * a loop in the DAG
         * @param from : edge's loader
         * @param to : edge'sloader
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