project(python-baylib)
cmake_minimum_required(VERSION 2.8)

#pybind11
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.6.1
)
FetchContent_MakeAvailable(pybind11)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

set(PYBIND11_LTO_CXX_FLAGS "")

if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

add_subdirectory(../baylib baylib_build)

set(binding_src
    bindings/bayesian_net.cpp
)

pybind11_add_module(_pybaylib ${binding_src})
target_link_libraries(_pybaylib PRIVATE baylib)

set_target_properties( _pybaylib
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        PREFIX ""
)

configure_file(${PROJECT_SOURCE_DIR}/baylib/__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/baylib/__init__.py)