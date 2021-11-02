cmake_minimum_required(VERSION 2.8 FATAL ERROR)
include(FetchContent)
FetchContent_Declare (
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG master
)

FetchContent_MakeAvailable(googlebenchmark)

macro(add_benchmark name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} benchmark::benchmark)
endmacro()

# add_benchmark(mybenchmark mybenchmark.cpp)
# ...