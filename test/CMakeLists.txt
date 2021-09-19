include(FetchContent)

# gtest
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
)


if (WIN32)
    # prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif (WIN32)

FetchContent_MakeAvailable(googletest)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()


macro(add_gtest name)
   add_executable(${name}_test ${name}_test.cpp)
   set_target_properties(${name}_test PROPERTIES OUTPUT_NAME "${name}")
   target_link_libraries(${name}_test gtest gtest_main baylib)
   add_test(${name}_test ${name})
endmacro()


add_gtest(bnet)
add_gtest(rank)
add_gtest(cpt)
add_gtest(parser)
add_gtest(cow)
add_gtest(inference)
add_gtest(evidence)
add_gtest(regression_inference)
