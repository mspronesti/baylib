//
// Created by elle on 22/07/21.
//

#ifndef BAYESIAN_INFERRER_COW_HPP
#define BAYESIAN_INFERRER_COW_HPP

#include <memory>


template <typename T>
class cow {
protected:
    void construct() {
        _ptr = std::make_shared<T>();
    }

    // call this function in derived's setter before other code
    void detach() {
        if(_ptr.use_count() > 1){
           std::shared_ptr<T> old = _ptr;
           construct();
           *_ptr = *old;
        }
    }


    const T * data() const{
        return _ptr.get();
    }

    T* data(){
        return _ptr.get();
    }

    long use_count() const {
        return _ptr.use_count();
    }

private:
    std::shared_ptr<T> _ptr;
};


#endif //BAYESIAN_INFERRER_COW_HPP
