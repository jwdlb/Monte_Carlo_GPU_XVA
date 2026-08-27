#include "gpu_lsm/cpu_paths.hpp"
#include "gpu_lsm/cva.hpp"
#include "gpu_lsm/lsm.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("unilateral CVA is non-negative and respects zero-risk limits") {
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 5'000;
    const gpu_lsm::BermudanPutParams option{};
    const auto paths = gpu_lsm::generate_exact_gbm_paths(gpu_lsm::MarketParams{}, 1.0, simulation);
    const auto lsm = gpu_lsm::price_bermudan_put_lsm(
        gpu_lsm::MarketParams{}, option, simulation, paths);
    auto credit = gpu_lsm::CreditParams{};
    const auto base = gpu_lsm::calculate_unilateral_cva(
        gpu_lsm::MarketParams{}, option, simulation, credit, lsm);
    REQUIRE(base.cva >= 0.0);
    REQUIRE(base.exposure.expected_exposure.size() == 53);
    auto higher_hazard = credit;
    higher_hazard.hazard_rate = 0.04;
    REQUIRE(gpu_lsm::calculate_unilateral_cva(
        gpu_lsm::MarketParams{}, option, simulation, higher_hazard, lsm).cva > base.cva);
    auto lower_recovery = credit;
    lower_recovery.recovery_rate = 0.20;
    REQUIRE(gpu_lsm::calculate_unilateral_cva(
        gpu_lsm::MarketParams{}, option, simulation, lower_recovery, lsm).cva > base.cva);
    credit.hazard_rate = 0.0;
    REQUIRE(gpu_lsm::calculate_unilateral_cva(
        gpu_lsm::MarketParams{}, option, simulation, credit, lsm).cva == Catch::Approx(0.0));
    credit = gpu_lsm::CreditParams{};
    credit.recovery_rate = 1.0;
    REQUIRE(gpu_lsm::calculate_unilateral_cva(
        gpu_lsm::MarketParams{}, option, simulation, credit, lsm).cva == Catch::Approx(0.0));
}

TEST_CASE("exposure is zero after a path exercises") {
    gpu_lsm::MarketParams market{};
    market.rate = 0.0;
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 2;
    const gpu_lsm::BermudanPutParams option{};
    const gpu_lsm::BermudanLsmResult policy{
        0.0, 0.0, 0.0, 0.0, {13, 52}, {10.0, 20.0}, {}};
    const auto result = gpu_lsm::calculate_unilateral_cva(
        market, option, simulation, gpu_lsm::CreditParams{}, policy);
    REQUIRE(result.exposure.expected_exposure[13] == Catch::Approx(15.0));
    REQUIRE(result.exposure.expected_exposure[14] == Catch::Approx(10.0));
}
