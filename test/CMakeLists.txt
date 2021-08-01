add_executable(bnet_test bnet_test.cpp)

set_target_properties( bnet_test PROPERTIES OUTPUT_NAME "bnet" )

# google test
find_package(GTest REQUIRED)

target_link_libraries(bnet_test gtest gtest_main pthread)

add_test( bnet_test bnet)
