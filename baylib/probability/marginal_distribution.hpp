//
// Created by elle on 15/08/21.
//

#ifndef BAYLIB_MARGINAL_DISTRIBUTION_HPP
#define BAYLIB_MARGINAL_DISTRIBUTION_HPP

#include <baylib/network/random_variable.hpp>

namespace bn {
    /**
     * This class models the marginal distribution
     * of a set of random variables.
     * Can be initialized using
     * - an iterable container containing bn::random_variables<Probability>
     * - two iterators of bn::random_variables<Probability>
     *
     * @tparam Probability  : the type expressing the probability
     */
    template <typename Probability>
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
                mdistr.emplace_back((*it).states().size(), 0.0);
        }

        void set(ulong vid, ulong state_value, Probability p) {
            BAYLIB_ASSERT(vid < mdistr.size() &&
                          state_value < mdistr[vid].size(),
                          "out of bound access to marginal "
                          "distribution",
                          std::out_of_range)

            BAYLIB_ASSERT(p >= 0.0 && p <= 1.0,
                          "Probability value " << p
                          << " ain't included in [0, 1]",
                          std::logic_error)

            mdistr[vid][state_value] = p;
        }

        std::vector<Probability> &operator[](ulong vid) {
            BAYLIB_ASSERT(vid < mdistr.size(),
                          "out of bound access to marginal "
                          "distribution",
                          std::out_of_range)

            return mdistr[vid];
        }

        void operator/=(Probability value) {
            for (auto &row : mdistr)
                for (auto &entry : row)
                    entry /= value;
        }

        marginal_distribution<Probability> &operator+=(
                const marginal_distribution<Probability> &other
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

        friend std::ostream &operator<<(
                std::ostream &os,
                const marginal_distribution<Probability> &md
        ) {
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
                Probability sum = std::accumulate(row.begin(), row.end(), 0.0);
                if (abs(sum) > 1.0e-5)
                    std::for_each(row.begin(), row.end(), [sum](auto &val) {
                        val /= sum;
                    });
            }
        }

    private:
        std::vector<std::vector<Probability>> mdistr;
    };

    // type deduction guide
    template<typename Iterator>
    marginal_distribution(Iterator begin, Iterator end) -> marginal_distribution<std::decay_t<decltype(*begin)>>;

} // namespace bn

#endif //BAYLIB_MARGINAL_DISTRIBUTION_HPP