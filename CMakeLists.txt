cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
include(CheckLanguage)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(baylib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH cmake_module)

# change this option to compile the examples
option(BUILD_EXAMPLES "Build the examples showing how to use baylib" OFF)

# change this option to compile tests
option(BUILD_TESTS "Build tests for baylib" OFF)

#CUDA
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Found CUDA installed on your system. Enabling CUDA support for baylib.")
    enable_language(CUDA)
    add_compile_definitions(BAYLIB_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
else()
    message(STATUS "CUDA was not found on your system. Baylib will not use CUDA.")
endif()

#openCL
if(NOT TARGET OpenCL::OpenCL)
    set(CMAKE_PREFIX_PATH "${OPENCL_ROOT}")
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "Found OpenCL installed on your system. Enabling OpenCL support for baylib.")
        set(BAYLIB_OPENCL ON)
        add_compile_definitions(BAYLIB_OPENCL)
        set_target_properties(OpenCL::OpenCL PROPERTIES INTERFACE_COMPILE_DEFINITIONS CL_TARGET_OPENCL_VERSION=220)
        add_compile_definitions(BAYLIB_OPENCL)
    else()
        message(STATUS "OpenCL was not found on your system. Baylib will not use OpenCL.")
        set(BAYLIB_OPENCL OFF)
    endif()
endif()


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
