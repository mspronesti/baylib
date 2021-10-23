//
// Created by elle on 05/08/21.
//

#ifndef BAYLIB_CONDITION_HPP
#define BAYLIB_CONDITION_HPP

#include <baylib/baylib_assert.h>

//! @file condition.hpp
//! @brief Condition class used for CPT indexing

namespace bn {
    using state_t = std::uint64_t;

    /**
     * Let X be a random variable, then
     * P(X | c) is the probability of  X given c.
     * This class models c and it is used  in the
     * conditional probability table
     */
    class condition {
    public:
        condition() = default;

        // copy constructor
        condition(const condition &c) {
            for(auto & [k,v] : c.cmap)
                cmap[k] = v;
        }

        /**
         * copy operator
         * @param other : condition to be copied
         * @return      : new condition
         */
        condition &operator=(const condition &other) {
            if (this == &other)
                return *this;

            condition(other).swap(*this);
            return *this;
        }

        /**
         * Add a new pair of id, value of node
         * @param node_id : id of node
         * @param val     : value of node
         */
        void add(const unsigned long node_id, state_t val) {
            cmap[node_id] = val;
        }

        /**
         * Operator to retrieve the state of a given node
         * @param node_id : node id
         * @return        : state
         */
        state_t &operator[](const unsigned long node_id) {
            BAYLIB_ASSERT(contains(node_id),
                    "condition doesn't contain node"
                    << node_id,
                    std::runtime_error)

            return cmap[node_id];
        }

        /**
        * Operator to retrieve the state of a given node
        * @param node_id : node id
        * @return        : state
        */
        const state_t &operator[](const unsigned long node_id) const {
            BAYLIB_ASSERT(contains(node_id),
                          "condition doesn't contain node"
                          << node_id,
                          std::runtime_error)

            return cmap.at(node_id);
        }

        /**
         * utility to swap content of two conditions
         * @param other : other condition
         */
        void swap(condition &other) {
            cmap.swap(other.cmap);
        }

        /**
         * empty the condition of its contents
         */
        void clear() {
            cmap.clear();
        }

        /**
         * check if a node was set in the condition
         * @param node_id : node id
         * @return  true if node_id corresponds to a set state in the condition
         */
        bool contains(const unsigned long node_id) const {
            return cmap.find(node_id) != cmap.end();
        }

        /**
         * begin iterator of condition contents
         * @return : iterator
         */
        auto begin() const {
            return cmap.begin();
        }

        /**
         * end iterator of condition contents
         * @return : iterator
         */
        auto end() const {
            return cmap.end();
        }

        /**
         * begin reverse iterator of condition contents
         * @return : iterator
         */
        auto rbegin() const {
            return cmap.rbegin();
        }

        /**
         * end reverse iterator of condition contents
         * @return : iterator
         */
        auto rend() const {
            return cmap.rend();
        }

        /**
         * get the number of set nodes in the condition
         * @return size of condition
         */
        unsigned int size() const {
            return cmap.size();
        }

        bool operator < ( const condition &other) const {
            if (cmap.size() < other.cmap.size())
                return true;
            else if (cmap.size() > other.cmap.size())
                return false;
            else {
                // compare keys and values
                auto it = other.cmap.begin();
                for (auto &[k, v] : cmap) {
                    if (v < it->second)
                        return true;
                    else if (v > it->second)
                        return false;
                    else  if (k < it->first)
                        return true;
                    else if (k > it->first)
                        return false;

                    it++;
                }
                return false;
            }
        }

        /**
         * operator for printing the condition on stream
         * @param os : stream
         * @param c  : condition
         * @return   : stream
         */
        friend std::ostream& operator << (std::ostream &os, const condition &c)
        {
          for(auto & [k,v] : c.cmap)
              os << k << ':' << v << ' ';
          return os;
        }


    private:
        // ! key   : node id
        // ! value : state value
        std::map<unsigned long, state_t> cmap;
    };
}
#endif //BAYLIB_CONDITION_HPP
