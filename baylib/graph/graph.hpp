//
// Created by elle on 01/08/21.
//

#ifndef BAYESIAN_INFERRER_GRAPH_HPP
#define BAYESIAN_INFERRER_GRAPH_HPP

#include <boost/graph/adjacency_list.hpp>
#include <baylib/probability/cpt.hpp>
#include <memory_resource>
namespace bn {
    template <typename Probability>
    struct random_variable {
        unsigned int id{};
        std::string name;
        bn::cow::cpt<Probability> cpt;
        std::vector<std::string> _states;
        
        random_variable() = default;

        random_variable(std::string name, const std::vector<std::string>& states)
            :name(std::move(name)), cpt(states.size()) {}

        bool has_state(const std::string &state_name){
            return std::any_of(_states.begin(), _states.end(), 
                    [state_name](std::string state){ return state_name == state; });
        }

        std::vector<std::string> states () const {
            return _states;
        }

        bn::cow::cpt<Probability> &table() {
            return cpt;
        }

        void set_probability(
          bn::state_t state_value,
          const bn::condition& cond,
          Probability p
        )
        {
            cpt.set_probability(cond, state_value, p);
        }

        bool operator < (const random_variable & other) const { return id < other.id; }
    };

    template <typename Probability>
    using graph = boost::adjacency_list<boost::listS,
                                        boost::vecS, // vertex list
                                        boost::bidirectionalS, // graph type
                                        random_variable<Probability> // vertex type
                                        >;

    template <typename Probability>
    using vertex = typename graph<Probability>::vertex_descriptor;


} // namespace bn

#endif //BAYESIAN_INFERRER_GRAPH_HPP
