set(CMAKE_CXX_STANDARD 17)
project(bayesian-inferrer)
cmake_minimum_required(VERSION 3.14)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# boost
set(Boost_DEBUG 1)
include_directories(${Boost_INCLUDE_DIR})
find_package(Boost REQUIRED)

set(threading=multi)

#openCL
find_package(OpenCL REQUIRED)
find_package (Threads REQUIRED)


find_package (Threads)
add_subdirectory(baylib)
add_subdirectory(test)
enable_testing()

add_executable(bayesian-inferrer main.cpp)

target_link_libraries(bayesian-inferrer OpenCL::OpenCL pthread boost_thread)


