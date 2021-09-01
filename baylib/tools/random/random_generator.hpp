//
// Created by paolo on 31/08/21.
//

#ifndef BAYLIB_RANDOM_GENERATOR_HPP
#define BAYLIB_RANDOM_GENERATOR_HPP

#define AVALANCHE_FACTOR 0x45D9F3B // from papers and experiments

namespace bn {
    /**
     * Random generator wrapper for sampling-based algorithms
     * Customization offered features:
     * - custom generator
     * - custom seed
     * - custom type for the random number
     * @tparam T           : the type of the random number
     * @tparam Generator   : the type of generator (default Marsenne Twister)
     */
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
          seed = (seed >> 16) ^ seed;
          return seed;
       }

       Generator gen;
    };

    template <typename T>
    class seed_factory {
    private:
    public:
        seed_factory(uint nseeds, uint seed = 0)
        : next(0)
        {
            std::seed_seq seq{seed};
            seeds = std::vector<ulong>(nseeds);
            seq.generate(seeds.begin(), seeds.end());
        }

        seed_factory(const seed_factory & other) = delete;
        seed_factory & operator = (const seed_factory & other) = delete;

        uint get_new () {
            BAYLIB_ASSERT(next < seeds.size(),
                          "seed factory already produced "
                          "the required number of seeds",
                          std::runtime_error)

            return seeds[next++];
        }

    private:
        std::vector<T> seeds;
        ulong next;
    };
}

#endif //BAYLIB_RANDOM_GENERATOR_HPP