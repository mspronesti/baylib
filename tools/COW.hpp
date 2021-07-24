//
// Created by elle on 22/07/21.
//

#ifndef BAYESIAN_INFERRER_COW_HPP
#define BAYESIAN_INFERRER_COW_HPP


#include <memory>

template <typename T>
class COW {
protected:
    void construct() {
        _ptr = std::make_shared<T>();
    }

    void clone_if_needed() {
        if(_ptr.use_count() > 1){
           std::shared_ptr<T> old = _ptr;
           construct();
           *_ptr = *old;
        }
    }

    void duplicate(const COW<T> &other) {
        _ptr = other._ptr;
    }

    const T * get() const {
        return _ptr.get();
    }

    long use_count() const {
        return _ptr.use_count();
    }

private:
    std::shared_ptr<T> _ptr;
};


#endif //BAYESIAN_INFERRER_COW_HPP
