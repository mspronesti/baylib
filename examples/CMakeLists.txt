cmake_minimum_required(VERSION 2.8 )
set(CMAKE_CXX_STANDARD 20)

project(examples)

macro(add_demo name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} baylib)
endmacro()

add_demo(example_inference)
add_demo(example_load_network)
add_demo(example_network)
add_demo(example_named_network)
add_demo(example_interact_network)
