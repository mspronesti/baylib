set(CMAKE_CXX_STANDARD 17)
project(bayesian-inferrer)
set(Boost_DEBUG 1)
include_directories(${Boost_INCLUDE_DIR})
find_package(Boost REQUIRED)
find_package(OpenCL REQUIRED)
find_package (Threads)
add_executable(bayesian-inferrer main.cpp parser/pugixml/pugixml.cpp old_graph/BayesianNet.cpp old_graph/BayesianNet.h old_graph/bnode.h test/testFile.hpp exceptions/FileNotFoundException.h
        inference/logic_sampling.hpp tools/COW.hpp tools/thread_pool.hpp parser/parser.h graphs/DAG.cpp graphs/DAG.h
        graphs/graphStructure/ArcSet.cpp graphs/graphStructure/ArcSet.h graphs/graphStructure/EdgeSet.cpp graphs/graphStructure/EdgeSet.h graphs/graphStructure/GraphElements.cpp
        networks/BayesianNet.h networks/io/BNReader.cpp networks/io/BNReader.h
        parser/rapidxml/rapidxml.hpp parser/rapidxml/rapidxml_iterators.hpp parser/rapidxml/rapidxml_print.hpp parser/rapidxml/rapidxml_utils.hpp)
target_link_libraries(bayesian-inferrer OpenCL::OpenCL pthread)
cmake_minimum_required(VERSION 3.14)


