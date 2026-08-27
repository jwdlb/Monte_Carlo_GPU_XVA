#pragma once

/*
 * Purpose: Declares statistical summaries used to turn many Monte Carlo
 * samples into a price estimate and an estimate of that price's uncertainty.
 */

#include "gpu_lsm/config.hpp"

#include <cstddef>
#include <span>

namespace gpu_lsm {

// Summary of a finite collection of samples, for example discounted payoffs.
struct SampleStatistics {
    std::size_t count{};
    double mean{};
    double sample_variance{};
    double standard_error{};
};

// Lower and upper bounds around an estimated quantity.
struct ConfidenceInterval {
    double low{};
    double high{};
};

// The first two moments of a model distribution at one time point.
struct DistributionMoments {
    double mean{};
    double variance{};
};

// Diagnostic comparison between simulated GBM moments and their known values.
struct MomentComparison {
    double time{};
    double sample_mean{};
    double theoretical_mean{};
    double relative_mean_error{};
    double sample_variance{};
    double theoretical_variance{};
    double relative_variance_error{};
};

// Calculates count, mean, sample variance, and standard error without copying.
[[nodiscard]] SampleStatistics summarize(std::span<const double> samples);

// Returns estimate +/- 1.95996... times the standard error.
[[nodiscard]] ConfidenceInterval confidence_interval_95(
    double estimate,
    double standard_error) noexcept;

// Returns the known risk-neutral mean and variance of geometric Brownian motion.
[[nodiscard]] DistributionMoments theoretical_gbm_moments(
    const MarketParams& market,
    double time);

// Compares a simulated time slice with the corresponding analytical GBM moments.
[[nodiscard]] MomentComparison compare_gbm_moments(
    std::span<const double> samples,
    const MarketParams& market,
    double time);

}  // namespace gpu_lsm
