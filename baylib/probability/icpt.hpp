//
// Created by paolo on 11/09/21.
//

#ifndef BAYLIB_ICPT_HPP
#define BAYLIB_ICPT_HPP

#include <probability/cpt.hpp>
#include <utility>

namespace bn{
    namespace cow{
        template <typename Probability>
        struct icpt: public bn::cow::cpt<Probability>{

            icpt()= default;

            icpt(const std::vector<ulong> &parents_size, uint states): icpt(parents_size, states, 1/static_cast<Probability>(states)){};

            explicit icpt(cow::cpt<Probability>& cpt, bool empty=false):
            bn::cow::cpt<Probability>(cpt){
                if(empty){
                    for(int i=0; i < this->size(); i++){
                        for(int j=0; j < (*this)[i].size(); j++)
                            (*this)[i][j] = 0;
                    }
                }
            }

            std::vector<Probability> &operator[] (const bn::condition &cond){
                return this->d->table[this->cond_map.at(cond)];
            }


            std::vector<Probability>& operator[](uint index){
                return this->d.data()->table[index];
            }

            const std::vector<Probability>& operator[](uint index) const{
                return this->d.data()->table[index];
            }

            void normalize(){
                for(std::vector<Probability>& row: this->d.data()->table){
                    Probability sum = std::accumulate(row.begin(), row.end(), 0.0);
                    std::transform(row.begin(), row.end(), row.begin(), [&sum](Probability& prob){return prob/sum;});
                }
            }

            /**
             * Merge two icpts using the formula this' = this + (other - this) * learning_rate
             * while returning the statistical distance between the two distributions
             * @param other
             * @param learning_rate
             * @return maximum mean discrepancy distance
             */
            double absorb(const icpt<Probability>& other, float learning_rate){
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
