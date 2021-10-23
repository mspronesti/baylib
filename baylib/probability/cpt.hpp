//
// Created by elle on 01/08/21.
//

#ifndef BAYLIB_CPT_HPP
#define BAYLIB_CPT_HPP


#include <baylib/probability/condition.hpp>
#include <numeric>
#include <vector>
#include <boost/functional/hash/hash.hpp>

#include <baylib/tools/cow/shared_data.hpp>
#include <baylib/tools/cow/shared_ptr.hpp>
#include <baylib/baylib_concepts.hpp>

/**
 * @file cpt.hpp
 * @brief CPT class with COW optimization
 */

/**
 * ========= Conditional Probability Table ==========
 * This class models the conditional probability table
 * associated to each variable of the bayesian network.
 * Employs custom shared_ptr from bn::cow namespace
 * adapted from Qt library source code to implement
 * copy-on-write
 */

namespace bn{
    template<RVarDerived Variable_>
    class bayesian_network;

    namespace cow {
        template<Arithmetic Probability_>
        struct CPTData : public bn::cow::shared_data {
            /**
             * "table" doesn't map condition into a probability
             * vector using a map because this struct is used for
             * copy-on-write
             * (only probability values are actually shared, regardless
             * of the condition entry so that COW is used for two tables
             * having the very same probability entries even if they refer
             * to different conditions )
             */
            std::vector<std::vector<Probability_>> table;
            unsigned long nstates{};
        };

        /**
        * This class models a condition probability
        * table indirectly mapping condition to probability
        * row (and employing copy on write to spare memory)
        *
        *  Example:
        *  bn::condition c = {{"var1": 1}, {"var2": 3}}
        *  bn::cow::cpt cpt{n}
        *  ...
        *  auto & probs = cpt[c]
        *
        *  probs[0] :  P(var3=0 | var1=1, var2=3)
        *  probs[1] :  P(var3=1 | var1=1, var2=3)
        *     .                 .
        *     .                 .
        *  probs[n] :  P(var3=n | var=1, var2=3)
        *  @tparam Probability_ : the type expressing probability.
        *                        Must be arithmetic
        **/
        template<Arithmetic Probability_>
        class cpt {
        public:
            explicit cpt(unsigned long nstates = 2) {
                d = new CPTData<Probability_>();
                d->nstates = nstates;
            }

            void set_probability(
                    const bn::condition &cond,
                    bn::state_t state_val,
                    Probability_ p
            ) {
                BAYLIB_ASSERT(state_val < d->nstates,
                              "invalid state value"
                              + std::to_string(state_val),
                              std::runtime_error)

                BAYLIB_ASSERT(p >= 0.0 && p <= 1.0,
                              "illegal probability value",
                              std::logic_error)

                if (has_entry_for(cond)) {
                    d->table[cond_map.at(cond)][state_val] = p;
                } else {
                    int size = d->table.size();
                    cond_map[cond] = size; // storing condition
                    d->table.emplace_back(d->nstates, 0.0); // alloccating new row in cpt
                    d->table[size][state_val] = p; // storing probability
                }
            }

            const std::vector<Probability_> &operator[] (const bn::condition &cond) const{
                BAYLIB_ASSERT(has_entry_for(cond),
                              "bad condition value",
                              std::out_of_range)

                 return d->table[cond_map.at(cond)];
            }


            const std::vector<Probability_>  &at(const bn::condition &cond) const{
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

            bool operator == (const cpt<Probability_> &c) const {
                return d->table == c.d->table;
            }

            unsigned long size() const {
                return d->table.size();
            }

            std::vector<Probability_> flat() const{
                auto flat_cpt = std::vector<Probability_>{};
                flat_cpt.reserve(d->table.size() * d->nstates);

                for(auto &[cond, cond_id] : cond_map) {
                    const auto cpt_row = d->table[cond_id];
                    flat_cpt.insert(flat_cpt.end(), cpt_row.begin(), cpt_row.end());
                }

                return flat_cpt;
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

            auto begin() const {
                return d->table.begin();
            }

            auto end() const {
                return d->table.end();
            }

            size_t hash(){
                const auto rows = d->table;
                size_t seed = 0;
                for(auto col: rows){
                    for(auto el: col){
                        auto const * p = reinterpret_cast<unsigned char const *>(&el) ;
                        int n = 1;
                        for(int i = 0; i < sizeof(Probability_) - 1; i++){
                            if(*(char *)&n == 1)
                                boost::hash_combine(seed, p[i]);
                            else
                                boost::hash_combine(seed, p[sizeof(Probability_) - 1 - i]);
                        }
                    }
                }
                return seed;
            }

            unsigned long number_of_states() const{ return d->nstates;}

            unsigned long number_of_conditions() const{ return d.data()->table.size();}


        protected:
            template <RVarDerived Variable_> friend class bn::bayesian_network;
            bn::cow::shared_ptr<CPTData<Probability_>> d;
            // assigns a condition its index in the cpt
            // ! key   : condition
            // ! value : row index
            std::map<bn::condition, unsigned long> cond_map;
        };


    } // namespace cow
} // namespace bn

#endif //BAYLIB_CPT_HPP
