#include "gpu_lsm/cpu_paths.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

TEST_CASE("exact GBM paths are reproducible and start at spot") {
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 500;
    simulation.num_time_steps = 52;
    const auto first = gpu_lsm::generate_exact_gbm_paths_with_timings(
        gpu_lsm::MarketParams{}, 1.0, simulation);
    const auto second = gpu_lsm::generate_exact_gbm_paths_with_timings(
        gpu_lsm::MarketParams{}, 1.0, simulation);
    REQUIRE(first.paths.num_times() == 53);
    REQUIRE(first.paths.num_paths() == 500);
    for (const double spot : first.paths.time_slice(0)) REQUIRE(spot == 100.0);
    for (std::size_t index = 0; index < first.paths.size(); ++index) {
        REQUIRE(std::isfinite(first.paths.data()[index]));
        REQUIRE(first.paths.data()[index] > 0.0);
        REQUIRE(first.paths.data()[index] == second.paths.data()[index]);
    }
    REQUIRE(first.timings.total_ms >= first.timings.path_evolution_ms);
}
