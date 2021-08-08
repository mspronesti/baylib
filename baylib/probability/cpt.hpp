//
// Created by elle on 01/08/21.
//

#ifndef BAYESIAN_INFERRER_CPT_HPP
#define BAYESIAN_INFERRER_CPT_HPP

#include <baylib/probability/condition.hpp>
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

namespace bn{
    namespace cow {

        template<typename Probability>
        struct CPTData : public bn::cow::shared_data {
            /**
             * "table" doesn't map condition into a probability
             * vector using a map because this struct is used for
             * copy-on-write
             * (only probability values are actually shared, regardless
             * of condition entry so that cow is used for two tables
             * having the very same probability entries of different
             *  conditions )
             */
            std::vector<std::vector<Probability>> table;
            unsigned int nstates{};
        };


        template<typename Probability>
        class cpt {
            /**
            * This class models a condition probability
            * table indirectly mapping condition to probability
            * row (and employing copy on write to spare memory)
            *
            *  Example:
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
        public:
            explicit cpt(unsigned int nstates = 2) {
                d = new CPTData<Probability>();
                d->nstates = nstates;
            }

            void set_probability(
                    const bn::condition &cond,
                    bn::state_t state_val,
                    Probability p
            ) {
                BAYLIB_ASSERT(state_val <= d->nstates,
                              "invalid state value",
                              std::runtime_error)

                BAYLIB_ASSERT(p >= 0.0 && p <= 1.0,
                              "illegal probability value",
                              std::logic_error)

                if (has_entry_for(cond)) {
                    d->table[cond_map.at(cond)][state_val] = p;
                } else {
                    int size = d->table.size();
                    cond_map[cond] = size; // storing condition
                    d->table.emplace_back(d->nstates, -1); // alloccating new row in cpt
                    d->table[size][state_val] = p; // storing probability
                }
            }

            std::vector<Probability> &operator[](const bn::condition &cond) {
                BAYLIB_ASSERT(has_entry_for(cond),
                        "bad condition value",
                        std::out_of_range)

                return d->table[cond_map.at(cond)];
            }

            std::vector<Probability> &at(const bn::condition &cond) {
                BAYLIB_ASSERT(has_entry_for(cond),
                              "bad condition value",
                              std::out_of_range)

                return d->table[cond_map.at(cond)];
            }

            std::vector<Probability> const &at(const bn::condition &cond) const {
                BAYLIB_ASSERT(has_entry_for(cond),
                              "bad condition value",
                              std::out_of_range)

                return d->table[cond_map.at(cond)];
            }

            void clear() {
                d->table.clear();
                cond_map.clear();
            }

            bool has_entry_for(const bn::condition &c) const {
                return cond_map.find(c) != cond_map.end();
            }

            bool operator == (const cpt<Probability> &c) const {
                // TODO: to be implemented
                // useful for cow assign
                return false;
            }

            friend std::ostream& operator << (std::ostream &os, const cpt &cpt) {
                for(auto &[cond, cond_id] : cpt.cond_map){
                    os << cond << " | ";
                    for(auto &p : cpt.d->table[cond_id])
                        os <<  ' ' << p << " | ";
                    os << '\n';
                }
                return os;
            }

        private:
            bn::cow::shared_ptr<CPTData<Probability>> d;
            // assigns a condition its index in the cpt
            // ! key   : condition
            // ! value : row index
            std::map<bn::condition, unsigned int> cond_map;
        };




    } // namespace cow
} // namespace bn
#endif //BAYESIAN_INFERRER_CPT_HPP
