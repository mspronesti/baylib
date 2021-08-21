set(CMAKE_CXX_STANDARD 17)
project(baylib)
cmake_minimum_required(VERSION 3.10)

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

add_subdirectory(baylib)
enable_testing()
add_subdirectory(test)

add_executable(bnetlib main.cpp)

target_link_libraries(bnetlib OpenCL::OpenCL pthread boost_thread Boost::system Boost::thread)