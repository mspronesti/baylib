//
// Created by elle on 05/08/21.
//

#ifndef BAYLIB_CONDITION_HPP
#define BAYLIB_CONDITION_HPP

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

        void add(const std::string &parent_name, state_t val) {
            // controllo ...

            cmap[parent_name] = val;
        }

        state_t &operator[](const std::string &parent_name) {
            if (!has_parent(parent_name)) {
                // throw ...
            }

            return cmap[parent_name];
        }

        const state_t &operator[](const std::string &parent_name) const {
            if (!has_parent(parent_name)) {
                // throw ...
            }

            return cmap.at(parent_name);
        }

        void swap(condition &other) {
            cmap.swap(other.cmap);
        }

        void clear() {
            cmap.clear();
        }

        bool has_parent(const std::string &state_name) const {
            return cmap.find(state_name) != cmap.end();
        }

        auto  begin() const {
            return cmap.begin();
        }

        auto end() const {
            return cmap.end();
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


    private:
        // ! key   : parent name
        // ! value : state value
        std::map<std::string, state_t> cmap;
    };
}
#endif //BAYLIB_CONDITION_HPP
