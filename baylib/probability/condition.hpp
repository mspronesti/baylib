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

    class condition {
        /**
         * Let X to be a random variable, then
         * P(X | c) is the probability of
         * X conditioned to "c".
         * This class models "c" and it is used
         * in the conditional probability table and
         * the joint probability table.
         */
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
         * Add a new pair of node name + value of node
         * @param node_name : name of node
         * @param val       : value of node
         */
        void add(const std::string &node_name, state_t val) {
            cmap[node_name] = val;
        }

        /**
         * Operator to retrieve the state of a given node
         * @param node_name : node name
         * @return          : state
         */
        state_t &operator[](const std::string &node_name) {
            BAYLIB_ASSERT(contains(node_name),
                    "condition doesn't contain node"
                    + node_name,
                    std::runtime_error)

            return cmap[node_name];
        }

        /**
        * Operator to retrieve the state of a given node
        * @param node_name : node name
        * @return          : state
        */
        const state_t &operator[](const std::string &node_name) const {
            BAYLIB_ASSERT(contains(node_name),
                          "condition doesn't contain node"
                          + node_name,
                          std::runtime_error)

            return cmap.at(node_name);
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
         * @param node_name : node name
         * @return          : true if node_name corresponds to a set state in the condition
         */
        bool contains(const std::string &node_name) const {
            return cmap.find(node_name) != cmap.end();
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
        // ! key   : node name
        // ! value : state value
        std::map<std::string, state_t> cmap;
    };
}
#endif //BAYLIB_CONDITION_HPP
