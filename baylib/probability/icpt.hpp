//
// Created by paolo on 11/09/21.
//

#ifndef BAYLIB_ICPT_HPP
#define BAYLIB_ICPT_HPP

#include <probability/cpt.hpp>
#include <utility>

/**
 * @file icpt.hpp
 * @brief ICPT class for learning CPTs distribution
 */


namespace baylib {
    namespace cow {
        /**
         * ICPT is a child class of CPT that enables learning of unknown distributions starting from
         * a known CPT or an empty table
         * @tparam Probability_ : type of ICPT entry
         */
        template <Arithmetic Probability_ = double >
        class icpt: public baylib::cow::cpt<Probability_>{

        public:
            icpt()= default;

            /**
             * icpt constructor, builds a new table given the cardinality of the parents and the number
             * of states of the variable, the distribution is automatically set to an uniform one
             * @param parents_size : vector of sizes of the parents
             * @param states       : number of states
             */
            explicit icpt(
                    const std::vector<ulong> &parents_size,
                    uint states
            )
            : icpt(parents_size, states, 1/static_cast<Probability_>(states))
            { };

            /**
             * constructor that builds an icpt given a cpt, optionally it empties the entries
             * @param cpt   : cpt that is copied
             * @param empty : if set true the icpt is filled with zeros
             */
            explicit icpt(
                    cow::cpt<Probability_>& cpt,
                    bool empty=false
            )
            : baylib::cow::cpt<Probability_>(cpt)
            {
                if(empty){
                    for(auto& row: this->d->table){
                        row = std::vector<Probability_>(row.size(), 0.);
                    }
                }
            }

            /**
             * operator that returns the distribution of a specific realization of the parents
             * @param cond : condition
             * @return     : probability distribution
            */
            std::vector<Probability_> &operator[] (const baylib::condition &cond){
                return this->d->table[this->cond_map.at(cond)];
            }


            std::vector<Probability_>& operator[](uint index){
                return this->d.data()->table[index];
            }

            /**
             * operator that returns the row of a cpt given the index
             * @param index : index
             * @return      : row of cpt
             */
            const std::vector<Probability_>& operator[](uint index) const{
                return this->d.data()->table[index];
            }

            /**
             * modifies the content of the icpt by dividing each row by its sum
             */
            void normalize(){
                for(std::vector<Probability_>& row: this->d.data()->table){
                    Probability_ sum = std::accumulate(row.begin(), row.end(), 0.0);
                    std::transform(row.begin(), row.end(), row.begin(), [&sum](Probability_& prob){return prob / sum;});
                }
            }

            /**
             * Merge two icpts using the formula this' = this + (other - this) * learning_rate
             * while returning the statistical distance between the two distributions
             * @param other
             * @param learning_rate
             * @return maximum mean discrepancy distance
             */
            double absorb(const icpt<Probability_>& other, float learning_rate){
                double tot_var_difference = 0.;
                for (int i = 0; i < this->size(); ++i) {
                    for(int j = 0; j < (*this)[i].size(); ++j){
                        double difference = other[i][j] - (*this)[i][j];
                        tot_var_difference = difference * difference;
                        (*this)[i][j] += learning_rate*(difference);
                    }
                }
                return tot_var_difference / (this->size()*(*this)[0].size());
            }

        };
    }
}


#endif //BAYLIB_ICPT_HPP
