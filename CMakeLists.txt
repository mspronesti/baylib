set(CMAKE_CXX_STANDARD 17)
project(bayesian-inferrer)
cmake_minimum_required(VERSION 3.14)

# boost
set(Boost_DEBUG 1)
include_directories(${Boost_INCLUDE_DIR})
find_package(Boost REQUIRED)

set(threading=multi)

#openCL
find_package(OpenCL REQUIRED)
find_package (Threads REQUIRED)


find_package (Threads)

# testing env
add_subdirectory(test)
enable_testing()

add_executable( bayesian-inferrer
        main.cpp
    #[[    parser/pugixml/pugixml.cpp
        parser/rapidxml/rapidxml.hpp
        parser/rapidxml/rapidxml_iterators.hpp
        parser/rapidxml/rapidxml_print.hpp
        parser/rapidxml/rapidxml_utils.hpp
        parser/BNReader.h]]
        inference/logic_sampling.hpp
        graph/graph.hpp
        network/bayesian_network.hpp
        network/probability/cpt.hpp
        tools/COW.hpp
        tools/thread_pool.hpp
        )

target_link_libraries(bayesian-inferrer OpenCL::OpenCL pthread boost_thread gtest gtest_main)


