//
// Created by elle on 01/08/21.
//

#ifndef BAYESIAN_INFERRER_CPT_HPP
#define BAYESIAN_INFERRER_CPT_HPP


#include <baylib/tools/COW.hpp>
#include <baylib/graph/graph.hpp>

namespace  bn {
    template <typename Probability>
    class cpt : public COW<Probability> {
        using COW<Probability>::construct;
        using COW<Probability>::ptr;
        using COW<Probability>::clone_if_needed;

    public:
        cpt(){
            construct();
            // TODO: to be implemented
        }

        std::vector<Probability> cpt_stripe() {
            // TODO: to be implemented
            return std::vector<Probability>{};
        }

        // other methods ...

    private:
        // matrix ...
        unsigned int nstates;
        std::vector<unsigned int> nstatesPerFather;
        std::vector<std::string>  realizations;
    };
} // namespace bn

#endif //BAYESIAN_INFERRER_CPT_HPP
