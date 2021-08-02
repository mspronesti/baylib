//
// Created by elle on 01/08/21.
//

#ifndef BAYESIAN_INFERRER_CPT_HPP
#define BAYESIAN_INFERRER_CPT_HPP


#include <baylib/tools/cow.hpp>
#include <baylib/graph/graph.hpp>
#include <numeric>

namespace  bn {
    template<typename Probability>
    using condition = std::map<bn::variable<Probability>, unsigned int>;

    template <typename Probability>
    using table_t = std::map<condition<Probability>, std::vector<Probability>>;

    template <typename Probability>
    struct table_data {
        table_t<Probability> table;
        std::vector<bn::variable<Probability>> parents;
    };

    template <typename Probability>
    class cpt : private cow<table_data<Probability>> {
        /**
         *  example:
         *  condition c = {{var1, 1}, {var2, 3}}
         *  cpt cpt(var3, {var1, var2}}
         *
         *  std::vector<Probability> probs = cpt[c]
         *
         *  probs[0] :  P(var3=0 | var1=1, var2=3)
         *  probs[1] :  P(var3=1 | var1=1, var2=3)
         *     .                 .
         *     .                 .
         *  probs[n] :  P(var3=n | var=1, var2=3)
         */
       using cow<table_data<Probability>>::construct;
       using cow<table_data<Probability>>::get;
       using cow<table_data<Probability>>::copy;

    public:
        explicit cpt(bn::variable<Probability> _owner) : _owner(_owner){
            construct();
            fill_table();
        }

        cpt(bn::variable<Probability> _owner, const std::vector<bn::variable<Probability>> &parents)
            : _owner(_owner)
        {
            construct();
            get()->_parents = std::move(parents);
            fill_table(parents);
        }

        std::vector<Probability> & operator [] (const condition<Probability> &cond){
            copy();
            return get()->table[cond];
        }

        std::vector<Probability> const& operator [] (const condition<Probability> &cond) const{
            return get()->table[cond];
        }

        std::vector<Probability> & at (const condition<Probability> &cond) {
            copy();
            return get()->table[cond];
        }

        std::vector<Probability> const& at (const condition<Probability> &cond) const{
            return get()->table[cond];
        }

        std::vector<bn::variable<Probability>> const &parents() const noexcept{
            return get()->parents;
        }

        void clear(){
            copy();
            // TODO: to be implemented
        }

        bn::vertex<Probability> owner() const {
            return _owner;
        }


    private:
        bn::variable<Probability> _owner; // the vertex the table belongs to
        void fill_table(const std::vector<bn::variable<Probability>> &parents = {});
    };

    template <typename Probability>
    void cpt<Probability>::fill_table(const std::vector<bn::variable<Probability>> &parents){
        if(parents.empty()){ // variable is a root
            unsigned int nstates = _owner.states.size();
            // controllo che non siano 0 ...
            auto cond = condition<Probability>{};
            auto probabilities = std::vector<Probability>{};

            for(int i = 0; i < nstates; ++i)
                probabilities.push_back(1.0/nstates);

            get()->table.insert(std::make_pair(cond, probabilities));
        }else {
            // TODO: to be implemented
        }
    }

} // namespace bn

#endif //BAYESIAN_INFERRER_CPT_HPP
