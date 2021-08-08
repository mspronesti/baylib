//
// Created by elle on 05/08/21.
//

#ifndef BAYLIB_CONDITION_HPP
#define BAYLIB_CONDITION_HPP

#include <baylib/assert.h>

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

        condition &operator=(const condition &other) {
            if (this == &other)
                return *this;

            condition(other).swap(*this);
            return *this;
        }

        void add(const std::string &node_name, state_t val) {
            cmap[node_name] = val;
        }

        state_t &operator[](const std::string &node_name) {
            BAYLIB_ASSERT(contains(node_name),
                    "condition doesn't contain node"
                    + node_name,
                    std::runtime_error)

            return cmap[node_name];
        }

        const state_t &operator[](const std::string &node_name) const {
            BAYLIB_ASSERT(contains(node_name),
                          "condition doesn't contain node"
                          + node_name,
                          std::runtime_error)

            return cmap.at(node_name);
        }

        void swap(condition &other) {
            cmap.swap(other.cmap);
        }

        void clear() {
            cmap.clear();
        }

        bool contains(const std::string &node_name) const {
            return cmap.find(node_name) != cmap.end();
        }

        auto  begin() const {
            return cmap.begin();
        }

        auto end() const {
            return cmap.end();
        }

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
                    if (k < it->first)
                        return true;
                    else if (k > it->first)
                        return false;
                    else if (v < it->second)
                        return true;
                    else if (v > it->second)
                        return false;

                    it++;
                }
                return false;
            }
        }

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
