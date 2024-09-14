cmake_minimum_required(VERSION 3.23)
project(simd_algorithm)

set(CMAKE_CXX_STANDARD 23)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(">> using Clang")
    add_compile_options(-mavx2)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(">> using GCC")
    add_compile_options(-mavx2)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(">> using MSVC")
    add_compile_options(/arch:AVX2)
endif ()

include_directories(${PROJECT_SOURCE_DIR})

add_executable(simd_algorithm
        #        test/simd_supported.cpp
        #        test/change_case.cpp
        #        test/minmax.cpp
        #        test/reverse.cpp
        #        test/is_sorted.cpp
        #        main.cpp
)
