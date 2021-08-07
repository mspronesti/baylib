//
// Created by elle on 01/08/21.
//

#ifndef BAYESIAN_INFERRER_CPT_HPP
#define BAYESIAN_INFERRER_CPT_HPP

#include <baylib/network/probability/condition.hpp>
#include <baylib/tools/cow.hpp>
#include <numeric>

#include <baylib/tools/cow/shared_data.hpp>
#include <baylib/tools/cow/shared_ptr.hpp>


namespace  bn {

    template <typename Probability>
    using table_t = std::map<bn::condition, std::vector<Probability>>;

    template <typename Probability>
    struct table_data {
        table_t<Probability> table;
        std::vector<std::string> states;
    };

    template <typename Probability>
class cpt : private ::cow<table_data<Probability>> {
        /**
         *  example:
         *  condition c = {{"var1": 1}, {"var2": 3}}
         *  cpt cpt({"var1", "var2"}}
         *
         *  std::vector<Probability> probs = cpt[c]
         *
         *  probs[0] :  P(var3=0 | var1=1, var2=3)
         *  probs[1] :  P(var3=1 | var1=1, var2=3)
         *     .                 .
         *     .                 .
         *  probs[n] :  P(var3=n | var=1, var2=3)
         */
       using ::cow<table_data<Probability>>::construct;
       using ::cow<table_data<Probability>>::data;
       using ::cow<table_data<Probability>>::detach;

    public:
        explicit cpt(const std::vector<std::string> &states = {"T", "F"})
        {
            construct();
            data()->states = states;
        }

        void set_probability (
            const bn::condition &cond,
            bn::state_t state_val,
            Probability p
         )
        {
            if(state_val > data()->states.size())
                throw std::runtime_error("invalid state value");

            if( p < 0.0 || p > 1.0)
                throw std::runtime_error("invalid probability value");

            detach();
            if(has_entry_for(cond)){
                data()->table[cond][state_val] = p;
            } else {
                auto tmp = std::vector<Probability>(data()->states.size(), -1);
                data()->table[cond] = std::move(tmp);
                data()->table[cond][state_val] = p;
            }
        }

        std::vector<Probability> & operator [] (const bn::condition &cond){
            detach();
            return data()->table[cond];
        }

        std::vector<Probability> & at (const bn::condition &cond) {
            detach();
            return data()->table[cond];
        }

        std::vector<Probability> const& at (const bn::condition &cond) const{
            return data()->table[cond];
        }

        std::vector<std::string> const &states() const noexcept{
            return data()->states;
        }

        void clear(){
            detach();
            data()->table.clear();
        }

        bool has_entry_for(const bn::condition &c) const{
            auto _table = data()->table;
            return _table.find(c) != _table.end();
        }

    private:
        bool coherent_table(){
            return false;
        }
    };

} // namespace bn




/**
 * ============== SECOND POSSIBILITY OF DEFINING IT =================
 * Using custom shared_ptr from bn::cow namespace adapted from
 * Qt library source code
 */
namespace bn::cow {
    template <typename Probability>
    struct CPTData : public bn::cow::shared_data {
        std::map<bn::condition, std::vector<Probability>> table;
        std::vector<std::string> states;
    };


    template<typename Probability>
    class cpt {
    public:
        explicit cpt(const std::vector<std::string> &states = {"T", "F"})
        {
            d = new CPTData<Probability>();
            d->states = states;
        }

        void set_probability (
                const bn::condition &cond,
                bn::state_t state_val,
                Probability p
        )
        {
            if(state_val > d->states.size())
                throw std::runtime_error("invalid state value");

            if( p < 0.0 || p > 1.0)
                throw std::runtime_error("invalid probability value");

            if(has_entry_for(cond)){
                d->table[cond][state_val] = p;
            } else {
                auto tmp = std::vector<Probability>(d->states.size(), -1);
                d->table[cond] = std::move(tmp);
                d->table[cond][state_val] = p;
            }
        }

        std::vector<Probability> & operator [] (const bn::condition &cond){
            return d->table[cond];
        }

        std::vector<Probability> & at (const bn::condition &cond) {
            return d->table[cond];
        }

        std::vector<Probability> const& at (const bn::condition &cond) const{
            return d->table[cond];
        }

        std::vector<std::string> const &states() const noexcept{
            return d->states;
        }

        void clear(){
            d->table.clear();
        }

        bool has_entry_for(const bn::condition &c) const{
            auto _table = d->table;
            return _table.find(c) != _table.end();
        }

    private:
        bn::cow::shared_ptr<CPTData<Probability>> d;
    };
}
#endif //BAYESIAN_INFERRER_CPT_HPP
