cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 20)

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
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

set(BAYLIB_REQUIRED_LIBS
    OpenCL
    pthread
    boost_thread
    boost_system
    tbb
)

set (src
    graph/graph.hpp
    inference/abstract_inference_algorithm.hpp
    inference/gibbs_sampling.hpp
    inference/likelihood_weighting.hpp
    inference/logic_sampling.hpp
    inference/rejection_sampling.hpp
    inference/adaptive_importance_sampling.hpp
    network/bayesian_net.hpp
    network/bayesian_utils.hpp
    network/random_variable.hpp
    probability/condition.hpp
    probability/condition_factory.hpp
    probability/cpt.hpp
    probability/icpt.hpp
    probability/marginal_distribution.hpp
    tools/cow/shared_data.hpp
    tools/cow/shared_ptr.hpp
    tools/gpu/gpu_utils.hpp
    tools/random/random_generator.hpp
    smile_utils/smile_utils.hpp
    baylib_assert.h
    baylib_concepts.hpp
)
set (src_cuda
        inference/cuda/samplers_cuda.cuh
        inference/cuda/samplers_cuda.cu
        inference/logic_sampling_cuda.hpp
        inference/likelihood_weighting_cuda.hpp
        tools/gpu/cuda_utils.cuh
        tools/gpu/cuda_utils.cu
    )

if(CMAKE_CUDA_COMPILER)
    list(APPEND src ${src_cuda})
endif()

set (BAYLIB_INCLUDE_LIBS ../baylib)

if(NOT TARGET baylib)
    add_library(baylib STATIC ${src})
endif()

set_target_properties(baylib PROPERTIES LINKER_LANGUAGE CXX)

if(CMAKE_CUDA_COMPILER)
    set_target_properties(baylib
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            LINKER_LANGUAGE CXX)
endif()

target_include_directories(baylib
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        INTERFACE $<INSTALL_INTERFACE:include>
        PUBLIC ${BAYLIB_INCLUDE_LIBS}
        )

target_link_libraries(baylib ${BAYLIB_REQUIRED_LIBS})

install(DIRECTORY ./
        DESTINATION include/baylib
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "*.cuh"
        PATTERN "*.cu"
)

export(TARGETS baylib FILE baylib-config.cmake)
set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE baylib)

