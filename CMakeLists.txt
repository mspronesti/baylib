project(gputest)
#add_compile_options(-lOpenCL)
# see https://cmake.org/cmake/help/latest/module/FindBoost.html
set(Boost_DEBUG 1)
include_directories(${Boost_INCLUDE_DIR})
find_package(Boost REQUIRED)


#find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)
#find_package(OpenCL REQUIRED)
#set(LIBS ${LIBS} ${Boost_LIBRARIES} ${OpenCL_LIBRARY})


#[[
include_directories( ${OPENCL_INCLUDE_DIR} )
include_directories(C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.4/include/ )
include_directories(C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.4/include/CL )


link_directories(C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.4/lib/x64/)
include_directories(D:/libraries/OpenCL-SDK/external/OpenCL-Headers)
link_directories(D:/libraries/OpenCL-SDK/external/OpenCL-ICD-Loader/loader)
link_directories(${Boost_LIBRARY_DIRS} ${OpenCL_LIBRARY})
]]
add_executable(BoostTest main.cpp pugixml/pugixml.cpp graph/BayesianNet.cpp graph/BayesianNet.h graph/bnode.cpp graph/bnode.h test/testFile.hpp)


cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

