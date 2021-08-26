set(CMAKE_CXX_STANDARD 17)
project(baylib)
cmake_minimum_required(VERSION 3.13)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(baylib)
enable_testing()
add_subdirectory(test)


