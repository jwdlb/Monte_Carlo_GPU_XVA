#include "gpu_lsm/black_scholes.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("normal CDF is centred at one half") {
    REQUIRE(gpu_lsm::normal_cdf(0.0) == Catch::Approx(0.5));
}

TEST_CASE("Black-Scholes put rejects invalid inputs") {
    gpu_lsm::MarketParams market{};
    market.spot = 0.0;
    REQUIRE_THROWS_AS(
        gpu_lsm::black_scholes_put(market, 100.0, 1.0),
        std::invalid_argument);
}

TEST_CASE("Black-Scholes default put matches the reference value") {
    SKIP("Enable after implementing black_scholes_put");
}

