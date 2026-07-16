cmake_minimum_required(VERSION 3.22)

project(
    gpu_lsm_xva
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated Longstaff-Schwartz Monte Carlo for XVA exposure"
    LANGUAGES CXX
)

option(GPU_LSM_BUILD_TESTS "Build the Catch2 test suite" OFF)

add_library(gpu_lsm_phase1
    src/black_scholes.cpp
    src/benchmark.cpp
    src/cpu_paths.cpp
    src/statistics.cpp
)

target_compile_features(gpu_lsm_phase1 PUBLIC cxx_std_20)
target_include_directories(
    gpu_lsm_phase1
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(gpu_lsm_phase1 PRIVATE /W4 /permissive-)
else()
    target_compile_options(gpu_lsm_phase1 PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(gpu_lsm_xva src/main.cpp)
target_link_libraries(gpu_lsm_xva PRIVATE gpu_lsm_phase1)

if(GPU_LSM_BUILD_TESTS)
    include(CTest)
    find_package(Catch2 3 REQUIRED)

    add_executable(phase1_tests
        tests/test_black_scholes.cpp
        tests/test_paths.cpp
        tests/test_statistics.cpp
    )
    target_link_libraries(phase1_tests PRIVATE gpu_lsm_phase1 Catch2::Catch2WithMain)

    include(Catch)
    catch_discover_tests(phase1_tests)
endif()
