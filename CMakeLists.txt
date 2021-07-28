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

# google test
find_package(GTest REQUIRED)

find_package (Threads)

add_executable(bayesian-inferrer main.cpp parser/pugixml/pugixml.cpp exceptions/FileNotFoundException.h
        inference/logic_sampling.hpp tools/COW.h tools/thread_pool.hpp graphs/DAG.cpp graphs/DAG.h
        graphs/graphStructure/ArcSet.cpp graphs/graphStructure/ArcSet.h graphs/graphStructure/EdgeSet.cpp graphs/graphStructure/EdgeSet.h graphs/graphStructure/GraphElements.cpp
        networks/BayesianNet.h
        parser/rapidxml/rapidxml.hpp parser/rapidxml/rapidxml_iterators.hpp parser/rapidxml/rapidxml_print.hpp parser/rapidxml/rapidxml_utils.hpp parser/BNReader.h
        tools/VariableNodeMap.cpp tools/VariableNodeMap.h probability/CPT.h)
target_link_libraries(bayesian-inferrer OpenCL::OpenCL pthread boost_thread)
add_executable( bayesian-inferrer
        main.cpp
        parser/pugixml/pugixml.cpp
        parser/rapidxml/rapidxml.hpp
        parser/rapidxml/rapidxml_iterators.hpp
        parser/rapidxml/rapidxml_print.hpp
        parser/rapidxml/rapidxml_utils.hpp
        parser/BNReader.h
        inference/logic_sampling.hpp
        tools/COW.h
        tools/thread_pool.hpp
        graph/DAG.cpp
        graph/DAG.h
        graph/structure/ArcSet.cpp
        graph/structure/ArcSet.h
        graph/structure/EdgeSet.cpp
        graph/structure/EdgeSet.h
        graph/structure/GraphElements.cpp
        network/bayesian_network.h
        tools/VariableNodeMap.cpp
        tools/VariableNodeMap.h
        probability/CPT.h )

target_link_libraries(bayesian-inferrer OpenCL::OpenCL gtest gtest_main pthread)


