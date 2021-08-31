cmake_minimum_required(VERSION 3.10.0)
project(examples)

add_executable(ex_inf example_inference.cpp)
add_executable(ex_parse example_load_network.cpp)
add_executable(ex_load example_create_network.cpp)
add_executable(ex_net example_interact_network.cpp)

target_link_libraries(ex_inf baylib)
target_link_libraries(ex_parse baylib)
target_link_libraries(ex_load baylib)
target_link_libraries(ex_net baylib)