add_executable(bnet_test bnet_test.cpp)
add_executable(rank_test rank_test.cpp)
add_executable(cpt_test  cpt_test.cpp)
add_executable(parser_test parser_test.cpp)
add_executable(logic_sampling_test logic_sampling_test.cpp)
add_executable(gibbs_sampling_test gibbs_sampling_test.cpp)

set_target_properties( bnet_test PROPERTIES OUTPUT_NAME "bnet" )
set_target_properties( rank_test PROPERTIES OUTPUT_NAME "rank" )
set_target_properties( cpt_test PROPERTIES OUTPUT_NAME "cpt" )
set_target_properties( parser_test PROPERTIES OUTPUT_NAME "parser")
set_target_properties( logic_sampling_test PROPERTIES OUTPUT_NAME "logic_sampling")
set_target_properties( gibbs_sampling_test PROPERTIES OUTPUT_NAME "gibbs_sampling")

# google test
find_package(GTest REQUIRED)

macro(add_gtest output_name)
   target_link_libraries(${output_name}_test gtest gtest_main baylib)
   add_test(${output_name}_test ${output_name})
endmacro()

add_gtest(bnet)
add_gtest(rank)
add_gtest(cpt)
add_gtest(parser)
add_gtest(logic_sampling)
add_gtest(gibbs_sampling)
