//
// Created by paolo on 31/08/21.
//

#ifndef BAYLIB_GENERATOR_WRAPPER_HPP
#define BAYLIB_GENERATOR_WRAPPER_HPP

template <typename T, typename Generator=std::mt19937>
struct generator_wrapper{

    using dist_type = typename std::conditional
            <
            std::is_integral<T>::value
            , std::uniform_int_distribution<T>
            , std::uniform_real_distribution<T>
            >::type;

    Generator gen;
    uint seed;

    explicit generator_wrapper(uint seed = 0): seed(seed), gen(seed){}

    T operator()(T from = .0, T to = 1.)
    {
        dist_type dist;
        return dist(gen, typename dist_type::param_type{from, to});
    }

};

#endif //BAYLIB_GENERATOR_WRAPPER_HPP
