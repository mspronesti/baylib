set(CMAKE_CXX_STANDARD 17)
project(baylib)
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
# change this option to compile the examples
option(BUILD_EXAMPLES "Build the examples showing how to use baylib" OFF)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(baylib)
enable_testing()
add_subdirectory(test)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
unset (BUILD_EXAMPLES CACHE)