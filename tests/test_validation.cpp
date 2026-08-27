#include "gpu_lsm/cpu_paths.hpp"
#include "gpu_lsm/lsm.hpp"
#include "gpu_lsm/validation.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Bermudan LSM passes European and American validation bounds") {
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 30'000;
    const gpu_lsm::BermudanPutParams option{};
    const auto paths = gpu_lsm::generate_exact_gbm_paths(gpu_lsm::MarketParams{}, 1.0, simulation);
    const auto lsm = gpu_lsm::price_bermudan_put_lsm(
        gpu_lsm::MarketParams{}, option, simulation, paths);
    const auto report = gpu_lsm::validate_bermudan_result(gpu_lsm::MarketParams{}, option, lsm);
    REQUIRE(report.european_put > 0.0);
    REQUIRE(report.american_put >= report.european_put);
    REQUIRE(report.passed);
}
