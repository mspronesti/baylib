//
// Created by elle on 07/08/21.
//

#ifndef BAYLIB_ASSERT_H
#define BAYLIB_ASSERT_H
#include <sstream>
#include <iosfwd>

#ifdef __GNUC__
#  if !(__GNUC__ == 4 && __GNUC_MINOR__ == 4 && __GNUC_PATCHLEVEL__ == 5)
#    define BAYLIB_FUNCTION_NAME __PRETTY_FUNCTION__
#  else
#    define BAYLIB_FUNCTION_NAME "unknown function"
#  endif
#elif defined(_MSC_VER)
#define BAYLIB_FUNCTION_NAME __FUNCSIG__
#else
#define BAYLIB_FUNCTION_NAME "unknown function"
#endif

#define BAYLIB_CASSERT(_exp, _msg, _except)                                 \
    {if ( !(_exp) )                                                         \
    {                                                                       \
        std::ostringstream osstr;                                       \
        osstr << "\n\nError detected at line " << __LINE__ << ".\n";    \
        osstr << "Error detected in file " << __FILE__ << ".\n";      \
        osstr << "Error detected in function " << BAYLIB_FUNCTION_NAME << ".\n\n";      \
        osstr << "Failing expression was " << #_exp << ".\n";           \
        osstr << std::boolalpha << _msg << "\n";                    \
        throw _except(osstr.str());      \
    }}

#define BAYLIB_ASSERT(...) BAYLIB_CASSERT(__VA_ARGS__)

#endif //BAYLIB_ASSERT_H
