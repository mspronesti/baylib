add_executable(bnet_test bnet_test.cpp cpt_test.cpp)
add_executable(rank_test rank_test.cpp)
add_executable(cpt_test cpt_test.cpp)

set_target_properties( bnet_test PROPERTIES OUTPUT_NAME "bnet" )
set_target_properties( rank_test PROPERTIES OUTPUT_NAME "rank" )
set_target_properties( cpt_test PROPERTIES OUTPUT_NAME "cpt" )

# google test
find_package(GTest REQUIRED)

target_link_libraries(bnet_test gtest gtest_main pthread)
target_link_libraries(rank_test gtest gtest_main pthread OpenCL::OpenCL)
target_link_libraries(cpt_test gtest gtest_main pthread)

add_test( bnet_test bnet)
add_test( rank_test rank)
add_test( cpt_test cpt)