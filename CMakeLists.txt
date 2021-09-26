cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(baylib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# change this option to compile the examples
option(BUILD_EXAMPLES "Build the examples showing how to use baylib" OFF)

# change this option to compile tests
option(BUILD_TESTS "Build tests for baylib" ON)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(baylib)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
unset (BUILD_TESTS CACHE)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
unset (BUILD_EXAMPLES CACHE)
