//
// Created by elle on 03/08/21.
//

#ifndef BAYESIAN_INFERRER_GRAPH2_HPP
#define BAYESIAN_INFERRER_GRAPH2_HPP

#include <iostream>
#include <map>
#include <vector>
#include <memory>
#include <boost/range/adaptor/map.hpp>
#include <boost/range/algorithm/copy.hpp>


/**
 * NOTA: mancano i controlli sulla mappa (da fare come fatti
 * nel bayesian_network usato fino ad ora)
 */
template<typename Probability>
class random_variable{
    using rv_ptr = std::shared_ptr<random_variable>;

    std::string _name;
    std::vector<std::string> _states;

    std::vector<rv_ptr> _parents;
    std::vector<rv_ptr> _children;

public:
    random_variable(std::string name, std::vector<std::string> states, int id=-1)
        : _name(std::move(name)), _states(std::move(states)) {}

    void add_parent(const rv_ptr &p) { _parents.push_back(p); }

    void add_child (const rv_ptr &c) { _children.push_back(c); }

    std::vector<std::string> states() const { return _states; }

    std::string name() const { return _name; }

    std::vector<rv_ptr> parents() { return _parents; }
    std::vector<rv_ptr> children() { return _children; }
};



template <typename Probability>
class bayesian_network {
    using rv_ptr = std::shared_ptr<random_variable<Probability>>;

    std::map<std::string, rv_ptr> vars;

public:
    bayesian_network() = default;

    bayesian_network(const bayesian_network &other) {
        // TODO: to be implemented
    }

    bayesian_network(bayesian_network && other){
        // TODO: to be implemented
    }

    bool is_root(const std::string &name){
        return vars.at(name)->parents().empty();
    }

    std::vector<rv_ptr> parents_of(const std::string &name){
        return vars.at(name)->parents();
    }

    std::vector<rv_ptr> children_of (const std::string &name){
           return vars.at(name)->children();
    }

    void add_variable(const std::string &name, const std::vector<std::string> &states){
        vars[name] = rv_ptr(new random_variable<Probability>{name, states});
    }

    void add_dependency(const std::string &name1, const std::string &name2){
        auto v1 = vars.at(name1);
        auto v2 = vars.at(name2);

        v1->add_child(v2);
        v2->add_parent(v1);
    }

    bool has_dependency(const std::string &name1, const std::string &name2){
        auto v1 = vars.at(name1);
        auto v2 = vars.at(name2);

        std::vector<rv_ptr> v = v1->children();

        return std::find(v.begin(), v.end(), v2) != v.end();
    }

    rv_ptr & operator [] (const std::string &name){
        return vars.at(name);
    }

    rv_ptr const & operator [] (const std::string &name) const {
        return vars.at(name);
    }

    std::vector<rv_ptr> variables(){
        auto variables = std::vector<rv_ptr>{};
        boost::copy(vars | boost::adaptors::map_keys, std::back_inserter(variables));

        return variables;
    }


};

#endif //BAYESIAN_INFERRER_GRAPH2_HPP
