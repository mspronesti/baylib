//
// Created by elle on 15/07/21.
//

#ifndef GPUTEST_FILENOTFOUNDEXCEPTION_H
#define GPUTEST_FILENOTFOUNDEXCEPTION_H

#include <exception>
#include <string>

class FileNotFoundException : public std::exception {

protected:
    std::string _msg;

public:
    explicit FileNotFoundException(const std::string &msg){
        _msg = msg;
    }

    const char* what() const noexcept override{
        return _msg.c_str();
    }

};


#endif //GPUTEST_FILENOTFOUNDEXCEPTION_H
