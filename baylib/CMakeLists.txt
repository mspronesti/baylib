cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# boost
if(NOT TARGET Boost::boost)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)
    set(Boost_DEBUG 1)
    find_package(Boost REQUIRED COMPONENTS system thread)
endif()

set(threading=multi)

#openCL
if(NOT TARGET OpenCL::OpenCL)
    set(CMAKE_PREFIX_PATH "${OPENCL_ROOT}")
    find_package(OpenCL REQUIRED)
    set_target_properties(OpenCL::OpenCL PROPERTIES INTERFACE_COMPILE_DEFINITIONS CL_TARGET_OPENCL_VERSION=220)
endif()
find_package (Threads REQUIRED)

set(required_external_libs
    OpenCL
    pthread
    boost_thread
    boost_system
    boost_thread
)

set (src
    graph/graph.hpp
    inference/gibbs_sampling.hpp
    inference/logic_sampling.hpp
    network/bayesian_network.hpp
    network/bayesian_utils.hpp
    network/random_variable.hpp
    parser/net_parser.hpp
    probability/condition.hpp
    probability/condition_factory.hpp
    probability/cpt.hpp
    probability/marginal_distribution.hpp
    tools/cow/shared_data.hpp
    tools/cow/shared_ptr.hpp
    tools/threads/thread_pool.hpp
    baylib_assert.h
)

set (required_includes ../baylib)

if(NOT TARGET baylib)
    add_library(baylib STATIC ${src})
endif()

set_target_properties(baylib PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(baylib PUBLIC "-std=c++17")

target_include_directories(baylib
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        INTERFACE $<INSTALL_INTERFACE:include>
        PUBLIC ${required_includes}
        )

target_link_libraries(baylib ${required_external_libs})

install(DIRECTORY ./
        DESTINATION include/baylib
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN "*.h"
)

export(TARGETS baylib FILE baylib-config.cmake)
set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE baylib)

