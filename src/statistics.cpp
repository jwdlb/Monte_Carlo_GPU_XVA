/*
 * Purpose: Implements stable statistical reporting for Monte Carlo samples and
 * analytical GBM moments used to validate simulated time slices.
 */

#include "gpu_lsm/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpu_lsm {

SampleStatistics summarize(std::span<const double> samples) {
    // This function is shared by Monte Carlo pricing and path diagnostics. It
    // returns the sample mean plus an uncertainty estimate without copying data.
    if (samples.empty()) {
        throw std::invalid_argument("Cannot summarize an empty sample");
    }

    // Welford's recurrence avoids the cancellation in E[X^2] - E[X]^2.
    std::size_t count = 0;
    double mean = 0.0;
    double sum_squared_deviations = 0.0;
    for (const double sample : samples) {
        if (!std::isfinite(sample)) {
            throw std::invalid_argument("Samples must all be finite");
        }
        ++count;
        const double delta = sample - mean;
        mean += delta / static_cast<double>(count);
        const double updated_delta = sample - mean;
        sum_squared_deviations += delta * updated_delta;
    }

    if (!std::isfinite(mean) || !std::isfinite(sum_squared_deviations)) {
        throw std::overflow_error("sample statistics overflowed");
    }

    const double sample_variance = count > 1
        ? std::max(sum_squared_deviations / static_cast<double>(count - 1), 0.0)
        : 0.0;
    const double standard_error =
        std::sqrt(sample_variance / static_cast<double>(count));
    if (!std::isfinite(sample_variance) || !std::isfinite(standard_error)) {
        throw std::overflow_error("sample statistics are not finite");
    }
    return {count, mean, sample_variance, standard_error};
}

ConfidenceInterval confidence_interval_95(
    double estimate,
    double standard_error) noexcept {
    // 97.5th standard-normal percentile: leaves 2.5% in each tail. This is a
    // normal-approximation interval for the Monte Carlo sample mean.
    constexpr double z_95 = 1.959963984540054;
    // A normal-approximation two-sided confidence interval centred on estimate.
    return {
        estimate - z_95 * standard_error,
        estimate + z_95 * standard_error,
    };
}

DistributionMoments theoretical_gbm_moments(
    const MarketParams& market,
    double time) {
    validate(market);
    if (!std::isfinite(time) || time < 0.0) {
        throw std::invalid_argument("time must be finite and non-negative");
    }

    // Closed-form moments let callers test whether generated GBM paths follow
    // the intended risk-neutral distribution.
    const double drift = market.rate - market.dividend_yield;
    const double mean = market.spot * std::exp(drift * time);
    const double variance =
        market.spot * market.spot * std::exp(2.0 * drift * time)
        * std::expm1(market.volatility * market.volatility * time);
    if (!std::isfinite(mean) || !std::isfinite(variance)) {
        throw std::overflow_error("theoretical GBM moments are not finite");
    }
    return {mean, std::max(variance, 0.0)};
}

MomentComparison compare_gbm_moments(
    std::span<const double> samples,
    const MarketParams& market,
    double time) {
    const auto sample = summarize(samples);
    const auto theoretical = theoretical_gbm_moments(market, time);

    const auto relative_error = [](double observed, double expected) {
        if (expected == 0.0) {
            return observed == 0.0
                ? 0.0
                : std::numeric_limits<double>::infinity();
        }
        return std::abs(observed - expected) / std::abs(expected);
    };

    return {
        time,
        sample.mean,
        theoretical.mean,
        relative_error(sample.mean, theoretical.mean),
        sample.sample_variance,
        theoretical.variance,
        relative_error(sample.sample_variance, theoretical.variance),
    };
}

}  // namespace gpu_lsm
