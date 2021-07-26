set(CMAKE_CXX_STANDARD 17)
project(bayesian-inferrer)
cmake_minimum_required(VERSION 3.14)

set(Boost_DEBUG 1)
set(threading=multi)

include_directories(${Boost_INCLUDE_DIR})
find_package(Boost REQUIRED)
find_package(OpenCL REQUIRED)
find_package (Threads)

add_executable(bayesian-inferrer main.cpp parser/pugixml/pugixml.cpp exceptions/FileNotFoundException.h
        inference/logic_sampling.hpp tools/COW.h tools/thread_pool.hpp graphs/DAG.cpp graphs/DAG.h
        graphs/graphStructure/ArcSet.cpp graphs/graphStructure/ArcSet.h graphs/graphStructure/EdgeSet.cpp graphs/graphStructure/EdgeSet.h graphs/graphStructure/GraphElements.cpp
        networks/BayesianNet.h
        parser/rapidxml/rapidxml.hpp parser/rapidxml/rapidxml_iterators.hpp parser/rapidxml/rapidxml_print.hpp parser/rapidxml/rapidxml_utils.hpp parser/BNReader.h
        tools/VariableNodeMap.cpp tools/VariableNodeMap.h probability/CPT.h)
target_link_libraries(bayesian-inferrer OpenCL::OpenCL pthread boost_thread)


