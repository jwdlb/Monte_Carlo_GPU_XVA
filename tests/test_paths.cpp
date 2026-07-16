#include "gpu_lsm/path_matrix.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("PathMatrix stores states in time-major order") {
    gpu_lsm::PathMatrix paths(3, 2);
    paths(1, 0) = 10.0;
    paths(1, 1) = 11.0;

    REQUIRE(paths.data()[2] == 10.0);
    REQUIRE(paths.data()[3] == 11.0);
    REQUIRE(paths.bytes() == 6 * sizeof(double));
}

TEST_CASE("PathMatrix rejects invalid dimensions and indices") {
    REQUIRE_THROWS_AS(gpu_lsm::PathMatrix(0, 10), std::invalid_argument);

    const gpu_lsm::PathMatrix paths(2, 2);
    REQUIRE_THROWS_AS(paths(2, 0), std::out_of_range);
    REQUIRE_THROWS_AS(paths(0, 2), std::out_of_range);
}

TEST_CASE("exact GBM path generation is reproducible") {
    SKIP("Enable after implementing generate_paths");
}

