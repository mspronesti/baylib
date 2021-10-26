//
// Created by elle on 19/10/21.
//

#ifndef BAYLIB_BAYLIB_CONCEPTS_HPP
#define BAYLIB_BAYLIB_CONCEPTS_HPP

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define __concepts_supported
#endif

#ifdef __concepts_supported
    #include <concepts>

    template < typename P_ >
    concept Arithmetic = std::is_arithmetic_v<P_>;

    // forward declaration of random_variable
    namespace baylib {
        template<Arithmetic>
        class random_variable;
    }

    template < typename V_ >
    concept RVarDerived = std::is_base_of_v< baylib::random_variable<typename V_::probability_type>, V_ >;

    // forward declaration of bayesian_net
    namespace baylib {
        template<RVarDerived>
        class bayesian_net;
    }

    template < typename N_ >
    concept BNetDerived = std::is_base_of_v< baylib::bayesian_net<typename N_::variable_type>, N_>;

    // to be compliant with std random engines,
    // the custom one must overload min(), max() and the
    // functional operator
    template < typename G_ >
    concept STDEngineCompatible = requires(G_ gen) {
        gen();
        gen.min();
        gen.max();
    };
#else
#define Arithmetic typename
#define RVarDerived typename
#define BNetDerived typename
#define STDEngineCompatible typename
#endif

#endif //BAYLIB_BAYLIB_CONCEPTS_HPP
