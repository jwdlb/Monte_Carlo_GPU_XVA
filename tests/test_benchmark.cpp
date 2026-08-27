#include "gpu_lsm/benchmark.hpp"
#include "gpu_lsm/cpu_paths.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("small workload reports finite path timing and memory") {
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 1'000;
    const auto result = gpu_lsm::generate_exact_gbm_paths_with_timings(
        gpu_lsm::MarketParams{}, 1.0, simulation);
    REQUIRE(result.timings.total_ms >= 0.0);
    REQUIRE(result.paths.bytes() > 0);
    REQUIRE(gpu_lsm::bytes_to_mebibytes(result.paths.bytes()) > 0.0);
}
