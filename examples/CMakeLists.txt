cmake_minimum_required(VERSION 3.10.0)
project(examples)

add_executable(ex example_xdsl.cpp)
target_link_libraries(ex baylib)