/*
 * Purpose: Tests the analytical normal-distribution building block and the
 * Black--Scholes European-put reference pricer.
 */

#include "gpu_lsm/black_scholes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <stdexcept>

// Symmetry of a standard normal distribution gives N(0) = 0.5.
TEST_CASE("normal CDF is centred at one half") {
    REQUIRE(gpu_lsm::normal_cdf(0.0) == Catch::Approx(0.5));
    REQUIRE(
        gpu_lsm::normal_cdf(-1.0)
        == Catch::Approx(1.0 - gpu_lsm::normal_cdf(1.0)));
}

// Validation must run before the currently unimplemented pricing formula.
TEST_CASE("Black-Scholes put rejects invalid inputs") {
    gpu_lsm::MarketParams market{};
    market.spot = 0.0;
    REQUIRE_THROWS_AS(
        gpu_lsm::black_scholes_put(market, 100.0, 1.0),
        std::invalid_argument);
}

TEST_CASE("Black-Scholes default put matches the reference value") {
    const double price = gpu_lsm::black_scholes_put(
        gpu_lsm::MarketParams{}, 100.0, 1.0);

    REQUIRE(price == Catch::Approx(5.573526022256971).epsilon(1e-12));
}

TEST_CASE("Black-Scholes put handles zero volatility") {
    auto market = gpu_lsm::MarketParams{};
    market.volatility = 0.0;

    const double expected =
        std::max(110.0 * std::exp(-market.rate) - market.spot, 0.0);
    REQUIRE(
        gpu_lsm::black_scholes_put(market, 110.0, 1.0)
        == Catch::Approx(expected).epsilon(1e-12));
}

TEST_CASE("Black-Scholes put behaves sensibly at extreme moneyness") {
    auto market = gpu_lsm::MarketParams{};
    market.spot = 1.0;
    REQUIRE(gpu_lsm::black_scholes_put(market, 100.0, 1.0) > 90.0);

    market.spot = 500.0;
    const double out_of_the_money =
        gpu_lsm::black_scholes_put(market, 100.0, 1.0);
    REQUIRE(out_of_the_money >= 0.0);
    REQUIRE(out_of_the_money < 1e-10);
}

TEST_CASE("Black-Scholes reports discount-factor overflow") {
    auto market = gpu_lsm::MarketParams{};
    market.rate = -1'000.0;
    REQUIRE_THROWS_AS(
        gpu_lsm::black_scholes_put(market, 100.0, 1.0),
        std::overflow_error);
}
