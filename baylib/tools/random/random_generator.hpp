//
// Created by paolo on 31/08/21.
//

#ifndef BAYLIB_RANDOM_GENERATOR_HPP
#define BAYLIB_RANDOM_GENERATOR_HPP
#define AVALANCHE_FACTOR 0x45D9F3B // from papers and experiments
namespace bn {
    template<typename T, typename Generator=std::mt19937>
    class random_generator {
        using dist_type = typename std::conditional
                <
                        std::is_integral<T>::value
                        , std::uniform_int_distribution<T>
                        , std::uniform_real_distribution<T>
                >::type;
    public:
        explicit random_generator(uint seed = 0) : gen(prime(seed)) {}

        T get_random(T from = .0, T to = 1.) {
            dist_type dist;
            return dist(gen, typename dist_type::param_type{from, to});
        }
    private:
       uint prime(uint seed)
       {
          seed = ((seed >> 16) ^ seed) * AVALANCHE_FACTOR;
          seed = ((seed >> 16) ^ seed) * AVALANCHE_FACTOR;
          seed = ((seed >> 16) ^ seed) * AVALANCHE_FACTOR;
          seed = (seed >> 16) ^ seed;
          return seed;
       }

       Generator gen;
    };
}

#endif //BAYLIB_RANDOM_GENERATOR_HPP
