//
// Created by elle on 07/08/21.
//

#ifndef BAYLIB_SHARED_DATA_HPP
#define BAYLIB_SHARED_DATA_HPP

#include <algorithm>
#include <atomic>
#include <utility>

/**
 * @file shared_data.hpp
 * @brief implementation of qshareddata.h from Qt library
 */

/**
 * Adapted version of qshareddata.h from Qt library
 * (which already implements the copy on write in an
 * efficient way)
 */
namespace baylib{
    namespace cow {
        class shared_data {
        public:
            mutable std::atomic<int> ref;

            inline shared_data() : ref(0) {}
            inline shared_data(const shared_data&) : ref(0) {}

            // using the assignment operator would lead to corruption in the ref-counting
            shared_data& operator = (const shared_data&) = delete;
        };
    } // namespace cow
} //namespace baylib

#endif //BAYLIB_SHARED_DATA_HPP
