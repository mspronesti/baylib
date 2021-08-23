//
// Created by elle on 15/08/21.
//

#ifndef BAYLIB_MARGINAL_DISTRIBUTION_HPP
#define BAYLIB_MARGINAL_DISTRIBUTION_HPP

#include <baylib/network/random_variable.hpp>

namespace bn {
    template <typename Probability>
    class marginal_distribution {
    public:
        explicit marginal_distribution(const std::vector<bn::random_variable<Probability>> &vars){
            for(ulong i = 0; i < vars.size(); ++i)
                mdistr.emplace_back(vars[i].states().size(), 0.0);
        }

        void set(ulong vid, ulong state_value, Probability p)
        {
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

        std::vector<Probability> & operator [] (ulong vid){
            BAYLIB_ASSERT(vid < mdistr.size(),
                          "out of bound access to marginal "
                          "distribution",
                          std::out_of_range)

            return mdistr[vid];
        }

        void operator /= (Probability value){
            for(auto & row : mdistr)
                for(auto & entry : row)
                    entry /= value;
        }

        friend std::ostream & operator << (
                std::ostream & os,
                const marginal_distribution<Probability> &md
        )
        {
            for(auto & row : md.mdistr){
                os <<  " | ";
                for(auto & p : row)
                    os <<  ' ' << p << " | ";
                os << '\n';
            }
            return os;
        }

    private:
        std::vector<std::vector<Probability>> mdistr;
    };
}

#endif //BAYLIB_MARGINAL_DISTRIBUTION_HPP
