#include "gpu_lsm/config.hpp"

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <stdexcept>

TEST_CASE("default Bermudan and credit configuration is valid") {
    const gpu_lsm::SimulationConfig simulation{};
    REQUIRE_NOTHROW(gpu_lsm::validate(gpu_lsm::MarketParams{}));
    REQUIRE_NOTHROW(gpu_lsm::validate(gpu_lsm::BermudanPutParams{}, simulation));
    REQUIRE_NOTHROW(gpu_lsm::validate(gpu_lsm::CreditParams{}));
}

TEST_CASE("Bermudan exercise schedule must be valid") {
    gpu_lsm::SimulationConfig simulation{};
    auto option = gpu_lsm::BermudanPutParams{};
    option.exercise_indices = {26, 13, 52};
    REQUIRE_THROWS_AS(gpu_lsm::validate(option, simulation), std::invalid_argument);
    option.exercise_indices = {13, 26};
    REQUIRE_THROWS_AS(gpu_lsm::validate(option, simulation), std::invalid_argument);
    option = gpu_lsm::BermudanPutParams{};
    option.strike = 0.0;
    REQUIRE_THROWS_AS(gpu_lsm::validate(option, simulation), std::invalid_argument);
}

TEST_CASE("credit validation rejects invalid values") {
    auto credit = gpu_lsm::CreditParams{};
    credit.hazard_rate = -0.01;
    REQUIRE_THROWS_AS(gpu_lsm::validate(credit), std::invalid_argument);
    credit = gpu_lsm::CreditParams{};
    credit.recovery_rate = 1.01;
    REQUIRE_THROWS_AS(gpu_lsm::validate(credit), std::invalid_argument);
    credit = gpu_lsm::CreditParams{};
    credit.hazard_rate = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(gpu_lsm::validate(credit), std::invalid_argument);
}
