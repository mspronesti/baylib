//
// Created by elle on 15/08/21.
//

#ifndef BAYLIB_MARGINAL_DISTRIBUTION_HPP
#define BAYLIB_MARGINAL_DISTRIBUTION_HPP

#include <baylib/network/random_variable.hpp>

/**
 * @file marginal_distribution.hpp
 * @brief output class of inference algorithms
 */


namespace baylib {
    /**
     * This class models the marginal distribution
     * of a set of random variables.
     * Can be initialized using
     * - an iterable container containing baylib::random_variables<probability_type>
     * - two iterators of baylib::random_variables<probability_type>
     *
     * @tparam Probability_  : the type expressing the probability
     */
    template <Arithmetic Probability_ = double >
    class marginal_distribution {
    public:
        template <typename Container>
        explicit marginal_distribution(const Container &vars){
            for(auto & var : vars)
                mdistr.emplace_back(var.states().size(), 0.0);
        }

        template<typename Iterator>
        marginal_distribution(Iterator begin, Iterator end) {
            for (auto it = begin; it != end; ++it)
                mdistr.emplace_back((*it).number_of_states(), 0.0);
        }

        void set(ulong vid, ulong state_value, Probability_ p) {
            BAYLIB_ASSERT(vid < mdistr.size() &&
                          state_value < mdistr[vid].size(),
                          "out of bound access to marginal "
                          "distribution",
                          std::out_of_range)

            BAYLIB_ASSERT(p >= 0.0 && p <= 1.0,
                          "probability_type value " << p
                          << " ain't included in [0, 1]",
                          std::logic_error)

            mdistr[vid][state_value] = p;
        }

        std::vector<Probability_> &operator[](ulong vid) {
            BAYLIB_ASSERT(vid < mdistr.size(),
                          "out of bound access to marginal "
                          "distribution",
                          std::out_of_range)

            return mdistr[vid];
        }

        void operator/=(Probability_ value) {
            for (auto &row : mdistr)
                for (auto &entry : row)
                    entry /= value;
        }

        marginal_distribution<Probability_> &operator+=(
                const marginal_distribution<Probability_> &other
        ) {
            BAYLIB_ASSERT(mdistr.size() == other.mdistr.size(),
                          "Incompatible second operand of type"
                          " marginal distribution",
                          std::logic_error)

            for (ulong i = 0; i < mdistr.size(); ++i) {
                BAYLIB_ASSERT(mdistr[i].size() == other.mdistr[i].size(),
                              "Incompatible second operand of type"
                              " marginal distribution",
                              std::logic_error)

                for (ulong j = 0; j < mdistr[i].size(); ++j)
                    mdistr[i][j] += other.mdistr[i][j];
            }
            return *this;
        }

        friend std::ostream &operator << (
                std::ostream &os,
                const marginal_distribution<Probability_> &md
        )
        {
            for (auto &row : md.mdistr) {
                os << " | ";
                for (auto &p : row)
                    os << ' ' << p << " | ";
                os << '\n';
            }
            return os;
        }

        void normalize() {
            for (auto &row : mdistr) {
                Probability_ sum = std::accumulate(row.begin(), row.end(), 0.0);
                if (abs(sum) > 1.0e-5)
                    std::for_each(row.begin(), row.end(), [sum](auto &val) {
                        val /= sum;
                    });
            }
        }

    private:
        std::vector<std::vector<Probability_>> mdistr;
    };

    // type deduction guide
    template<typename Iterator>
    marginal_distribution(Iterator begin, Iterator end) -> marginal_distribution<std::decay_t<decltype(*begin)>>;

} // namespace baylib

#endif //BAYLIB_MARGINAL_DISTRIBUTION_HPP