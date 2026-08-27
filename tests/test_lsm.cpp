#include "gpu_lsm/cpu_paths.hpp"
#include "gpu_lsm/lsm.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

TEST_CASE("LSM returns finite Bermudan price and valid exercise policy") {
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 20'000;
    const gpu_lsm::BermudanPutParams option{};
    const auto paths = gpu_lsm::generate_exact_gbm_paths(
        gpu_lsm::MarketParams{}, option.maturity, simulation);
    const auto result = gpu_lsm::price_bermudan_put_lsm(
        gpu_lsm::MarketParams{}, option, simulation, paths);
    REQUIRE(std::isfinite(result.price));
    REQUIRE(result.price >= 0.0);
    REQUIRE(result.standard_error >= 0.0);
    REQUIRE(result.exercise_indices.size() == simulation.num_paths);
    for (const auto index : result.exercise_indices) {
        REQUIRE((index == 13 || index == 26 || index == 39 || index == 52));
    }
}
