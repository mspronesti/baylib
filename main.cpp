//
// Created by paolo on 04/07/2021.
//
#include <iostream>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/network/probability/cpt.hpp>

/*void print_results(const std::vector<std::string>& names, std::vector<int> results){
    for (int i = 0; i < names.size(); ++i)
        std::cout << names[i] << ": " << results[i] << '\n';
}


void test_simulation_chain(){
    // COMA Network
    int n = 10000;
    auto bn = std::make_shared<bn::bayesian_network<float>>();
    auto ls = bn::logic_sampling(bn);

    std::pair<int, int> res_pair;
    std::vector<float> prob_source = {0.2, 0.8};
    std::vector<std::shared_ptr<bn::bcvec>> parent_res = {};
    std::shared_ptr<bn::bcvec> res = ls.simulate_node(prob_source, parent_res,  n);
    res_pair = ls.compute_result_binary(*res);
    std::cout << "MetastCancer:" << res_pair.first << " " << res_pair.second << "\n";
    prob_source = {0.8, 0.2, 0.2, 0.8};
    std::shared_ptr<bn::bcvec> res2 = ls.simulate_node(prob_source, {res}, n);
    res_pair = ls.compute_result_binary(*res2);
    std::cout << "IncreasedSerumCalcium:" << res_pair.first << " " << res_pair.second << "\n";
    prob_source = {0.2, 0.08, 0.05, 0.95};
    std::shared_ptr<bn::bcvec> res3 = ls.simulate_node(prob_source, {res}, n);
    res_pair = ls.compute_result_binary(*res3);
    std::cout << "BrainTumor:" << res_pair.first << " " << res_pair.second << "\n";
    prob_source = {0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.05, 0.095};
    std::shared_ptr<bn::bcvec> res4 = ls.simulate_node(prob_source, {res2, res3}, n);
    res_pair = ls.compute_result_binary(*res4);
    std::cout << "COMA:" << res_pair.first << " " << res_pair.second << "\n";
}

void test_simulation_non_binary(){
    // Animals Network
    int n = 100;
    auto bn = std::make_shared<bn::bayesian_network<float>>();
    auto ls = bn::logic_sampling(bn);

    auto res = ls.simulate_node({0.2, 0.2, 0.2, 0.2, 0.2}, {}, n, 5);
    auto acc_res = ls.compute_result_general(*res);
    std::vector<std::string> animals = {"Monkey", "Penguin", "Platypus", "Robin", "Turtle"};
    for (int i = 0; i < 5; i++)
        std::cout << animals[i] << ": " << acc_res[i] << '\n';
    res = ls.simulate_node({0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1}, {res}, n, 3);
    acc_res = ls.compute_result_general(*res);
    print_results({"Bird", "Mammal", "Reptile"}, acc_res);
}

void test_simulation_non_binary_multiple_parents(){
    // VentureBNExpanded Network
    int n = 10000;
    auto bn = std::make_shared<bn::bayesian_network<float>>();
    auto ls = bn::logic_sampling(bn);

    auto success_of_venture = ls.simulate_node({0.2, 0.8}, {}, n, 2);
    auto state_of_economy = ls.simulate_node({0.2, 0.7, 0.1}, {}, n, 3);
    auto expert_forecast = ls.simulate_node({0.7, 0.2, 0.1, 0.2, 0.3, 0.5, 0.6, 0.3, 0.1, 0.1, 0.3, 0.6, 0.5, 0.3, 0.2, 0.2, 0.4, 0.4},
                                            {success_of_venture, state_of_economy},
                                            n, 3);
    auto acc_res = ls.compute_result_general(*success_of_venture);
    print_results({"Success", "Failure"}, acc_res);
    acc_res = ls.compute_result_general(*state_of_economy);
    print_results({"Up", "Flat", "Down"}, acc_res);
    acc_res = ls.compute_result_general(*expert_forecast);
    print_results({"Good", "Moderate", "Poor"}, acc_res);

}*/

#include <baylib/graph/graph2.hpp>
#include <baylib/network/probability/condition.hpp>

int main(){
  /*  bayesian_network<double> g;
    g.add_variable("a", {});
    g.add_variable("b", {});
    g.add_variable("c", {});

    g.add_dependency("a", "b");
    std::cout << g.has_dependency("a", "b") << '\n'
              << g.is_root("a") << '\n'
              << g.has_dependency("b", "a");

*/
    bn::condition c{};
    c.add("a", 1);
    c.add("b", 1);

/*
    std::cout << c["a"] << '\t' << c["b"] << '\n';
*/

    bn::condition c1{};
    c1.add("c", 2);
    c1.add("a", 1);
  /*  std::cout << c["a"] << '\t' << c["b"] << '\n';
    std::cout << c1["c"] << '\n';
    std::cout << c["a"] << '\t' << c["b"] << '\n';*/


    std::map<bn::condition, int> m;
    m[c] = 1;
    std::cout << "c is present: " << (m.find(c) != m.end()) << '\n';
    m[c1] = 2;
    std::cout << "c1 is present: " << (m.find(c1) != m.end()) << '\n';


    std::cout << "---\n";

    for(auto & v : m)
        std::cout << v.second << '\n';


    std::cout << m.size();

    //test_simulation_chain();
    //test_simulation_non_binary_multiple_parents();
}