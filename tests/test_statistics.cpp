/*
 * Purpose: Tests stable sample summaries, confidence intervals, and analytical
 * geometric-Brownian-motion moments.
 */

#include "gpu_lsm/statistics.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

// A normal-approximation interval must be equally far above and below estimate.
TEST_CASE("95 percent interval is symmetric") {
    const auto interval = gpu_lsm::confidence_interval_95(10.0, 2.0);
    REQUIRE((10.0 - interval.low) == Catch::Approx(interval.high - 10.0));
}

TEST_CASE("sample statistics match a small known sample") {
    const std::array samples{1.0, 2.0, 3.0, 4.0};
    const auto statistics = gpu_lsm::summarize(samples);

    REQUIRE(statistics.count == 4);
    REQUIRE(statistics.mean == Catch::Approx(2.5));
    REQUIRE(statistics.sample_variance == Catch::Approx(5.0 / 3.0));
    REQUIRE(
        statistics.standard_error
        == Catch::Approx(std::sqrt((5.0 / 3.0) / 4.0)));
}

TEST_CASE("sample statistics handle one observation") {
    const std::array samples{7.5};
    const auto statistics = gpu_lsm::summarize(samples);

    REQUIRE(statistics.count == 1);
    REQUIRE(statistics.mean == 7.5);
    REQUIRE(statistics.sample_variance == 0.0);
    REQUIRE(statistics.standard_error == 0.0);
}

TEST_CASE("sample statistics remain stable for a large offset") {
    constexpr double offset = 1e12;
    const std::array samples{
        offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0};
    const auto statistics = gpu_lsm::summarize(samples);

    REQUIRE(statistics.mean == Catch::Approx(offset + 2.5));
    REQUIRE(statistics.sample_variance == Catch::Approx(5.0 / 3.0));
}

TEST_CASE("sample statistics reject unusable samples") {
    const std::array<double, 0> empty{};
    REQUIRE_THROWS_AS(gpu_lsm::summarize(empty), std::invalid_argument);

    const std::array non_finite{
        1.0, std::numeric_limits<double>::quiet_NaN()};
    REQUIRE_THROWS_AS(gpu_lsm::summarize(non_finite), std::invalid_argument);

    const std::array overflowing{
        std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max()};
    REQUIRE_THROWS_AS(gpu_lsm::summarize(overflowing), std::overflow_error);
}

TEST_CASE("theoretical GBM moments match their analytical formula") {
    const gpu_lsm::MarketParams market{};
    const auto moments = gpu_lsm::theoretical_gbm_moments(market, 1.0);
    const double expected_mean = market.spot * std::exp(market.rate);
    const double expected_variance =
        market.spot * market.spot * std::exp(2.0 * market.rate)
        * std::expm1(market.volatility * market.volatility);

    REQUIRE(moments.mean == Catch::Approx(expected_mean).epsilon(1e-12));
    REQUIRE(
        moments.variance
        == Catch::Approx(expected_variance).epsilon(1e-12));
}

TEST_CASE("GBM moment comparison handles the initial deterministic state") {
    const std::array samples{100.0, 100.0, 100.0};
    const auto comparison =
        gpu_lsm::compare_gbm_moments(samples, gpu_lsm::MarketParams{}, 0.0);

    REQUIRE(comparison.sample_mean == 100.0);
    REQUIRE(comparison.theoretical_mean == 100.0);
    REQUIRE(comparison.relative_mean_error == 0.0);
    REQUIRE(comparison.sample_variance == 0.0);
    REQUIRE(comparison.theoretical_variance == 0.0);
    REQUIRE(comparison.relative_variance_error == 0.0);
}

TEST_CASE("theoretical GBM moments reject invalid time") {
    REQUIRE_THROWS_AS(
        gpu_lsm::theoretical_gbm_moments(gpu_lsm::MarketParams{}, -0.5),
        std::invalid_argument);
}
