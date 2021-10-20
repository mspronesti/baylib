//
// Created by elle on 19/10/21.
//

#ifndef BAYLIB_BAYLIB_CONCEPTS_HPP
#define BAYLIB_BAYLIB_CONCEPTS_HPP

#include <concepts>

template < typename P_ >
concept Arithmetic = std::is_arithmetic_v<P_>;

// forward declaration of random_variable
namespace bn {
    template<Arithmetic>
    class random_variable;
}

template < typename V_ >
concept RVarDerived = std::is_base_of_v< bn::random_variable<typename V_::probability_type>, V_ >;

// forward declaration of bayesian_network
namespace bn {
    template<RVarDerived>
    class bayesian_network;
}

template < typename N_ >
concept BNetDerived = std::is_base_of_v< bn::bayesian_network<typename N_::variable_type>, N_>;

#endif //BAYLIB_BAYLIB_CONCEPTS_HPP
