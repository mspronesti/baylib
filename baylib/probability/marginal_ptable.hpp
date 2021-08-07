//
// Created by elle on 04/08/21.
//

#ifndef BAYESIAN_INFERRER_MARGINAL_PTABLE_HPP
#define BAYESIAN_INFERRER_MARGINAL_PTABLE_HPP

#include <vector>
#include <ostream>

namespace bn {
    template<typename Probability>
    class marginal_ptable {
    public:
        marginal_ptable(unsigned int nvars, const std::vector<bn::state_t> &states){
            mptable.resize(nvars, states);
        }

        void set(unsigned int var_id, unsigned int state_id, Probability p)
        {
            if(var_id >= mptable.size() || state_id >= mptable[var_id].size())
                throw std::out_of_range("marginal table out of range");

            mptable[var_id][state_id] = p;
        }

        void set(unsigned int var_id, const std::vector<Probability> probabilities)
        {
            if(var_id >= mptable.size())
                throw std::out_of_range("marginal table out of range");

            mptable[var_id] = probabilities;
        }

        std::vector<Probability>& operator [](unsigned int index) {
            if(index >= mptable.size())
                throw std::out_of_range("marginal table out of range");

            return mptable[index];
        }

    private:
        std::vector<std::vector<Probability>> mptable;
    };

    template <typename Probability>
    std::ostream& operator << (std::ostream &os, const marginal_ptable<Probability> &mpt){
        // TODO: to be implemented
        return os;
    }
}

#endif //BAYESIAN_INFERRER_MARGINAL_PTABLE_HPP
