set(CMAKE_CXX_STANDARD 17)
project(baylib)
cmake_minimum_required(VERSION 3.10)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# boost
find_package(Boost COMPONENTS system thread REQUIRED)
set(Boost_DEBUG 1)
include_directories(${Boost_INCLUDE_DIR})
set(threading=multi)

#openCL
find_package(OpenCL REQUIRED)
find_package (Threads REQUIRED)

add_subdirectory(baylib)


enable_testing()
add_subdirectory(test)

add_executable(bnetlib main.cpp)

target_link_libraries(bnetlib OpenCL::OpenCL pthread boost_thread Boost::system Boost::thread)