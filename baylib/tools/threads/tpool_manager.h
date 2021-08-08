//
// Created by elle on 02/08/21.
//

#ifndef BAYESIAN_INFERRER_TPOOL_MANAGER_H
#define BAYESIAN_INFERRER_TPOOL_MANAGER_H

#include <thread>

class tpool_manager {
public:
    static void initialize(unsigned int n = std::thread::hardware_concurrency());
    // other methods here ...
private:

};


#endif //BAYESIAN_INFERRER_TPOOL_MANAGER_H
