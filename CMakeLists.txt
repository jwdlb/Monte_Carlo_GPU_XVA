# Purpose: Defines the C++20 library, demonstration executable, and optional tests.
cmake_minimum_required(VERSION 3.22)

project(
    gpu_lsm_xva
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated Longstaff-Schwartz Monte Carlo for XVA exposure"
    LANGUAGES CXX
)

# Tests can be disabled for an offline/library-only build.
option(GPU_LSM_BUILD_TESTS "Build the Catch2 test suite" ON)
option(GPU_LSM_ENABLE_CUDA "Build the optional CUDA backend" OFF)

# The reusable CPU Bermudan-LSM/CVA reference library.
add_library(gpu_lsm_cpu
    src/black_scholes.cpp
    src/benchmark.cpp
    src/cpu_paths.cpp
    src/cva.cpp
    src/lsm.cpp
    src/statistics.cpp
    src/validation.cpp
)

# Consumers of the library must also compile with C++20.
target_compile_features(gpu_lsm_cpu PUBLIC cxx_std_20)
target_include_directories(
    gpu_lsm_cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Use useful warnings on the active compiler family.
if(MSVC)
    target_compile_options(gpu_lsm_cpu PRIVATE /W4 /permissive-)
else()
    target_compile_options(gpu_lsm_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# CUDA is optional: a CPU-only build remains valid on development machines and
# CI runners with no toolkit.  A CUDA build replaces the stub implementation.
set(GPU_LSM_HAS_CUDA OFF)
if(GPU_LSM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_library(gpu_lsm_cuda cuda/gpu_pipeline.cu)
        target_compile_features(gpu_lsm_cuda PUBLIC cxx_std_20)
        set_target_properties(gpu_lsm_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
        target_include_directories(gpu_lsm_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(gpu_lsm_cuda PUBLIC gpu_lsm_cpu CUDA::cudart)
        target_compile_definitions(gpu_lsm_cuda PUBLIC GPU_LSM_HAS_CUDA=1)
        set_property(TARGET gpu_lsm_cuda PROPERTY CUDA_ARCHITECTURES 70)
        set(GPU_LSM_HAS_CUDA ON)
    else()
        message(STATUS "CUDA compiler not found; building CPU fallback only")
    endif()
endif()

# Small command-line demonstration linked against the reusable library.
add_executable(gpu_lsm_xva src/main.cpp)
target_link_libraries(gpu_lsm_xva PRIVATE gpu_lsm_cpu)
if(GPU_LSM_HAS_CUDA)
    target_link_libraries(gpu_lsm_xva PRIVATE gpu_lsm_cuda)
else()
    target_sources(gpu_lsm_xva PRIVATE src/cuda_backend_stub.cpp)
endif()

# Catch2 is found locally when possible; otherwise CMake fetches the pinned release.
if(GPU_LSM_BUILD_TESTS)
    include(CTest)
    find_package(Catch2 3.15 QUIET)

    if(NOT Catch2_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.15.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    endif()

    # All CPU reference tests are built into one Catch2 test executable.
    add_executable(bermudan_cva_tests
        tests/test_black_scholes.cpp
        tests/test_config.cpp
        tests/test_lsm.cpp
        tests/test_paths.cpp
        tests/test_statistics.cpp
        tests/test_cva.cpp
        tests/test_benchmark.cpp
        tests/test_validation.cpp
    )
    target_link_libraries(bermudan_cva_tests PRIVATE gpu_lsm_cpu Catch2::Catch2WithMain)
    if(GPU_LSM_HAS_CUDA)
        target_sources(bermudan_cva_tests PRIVATE tests/test_cuda_backend.cpp)
        target_link_libraries(bermudan_cva_tests PRIVATE gpu_lsm_cuda)
    endif()

    include(Catch)
    catch_discover_tests(bermudan_cva_tests)
endif()
