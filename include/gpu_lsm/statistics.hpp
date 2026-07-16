#pragma once

#include <cstddef>
#include <span>

namespace gpu_lsm {

struct SampleStatistics {
    std::size_t count{};
    double mean{};
    double sample_variance{};
    double standard_error{};
};

struct ConfidenceInterval {
    double low{};
    double high{};
};

struct MomentComparison {
    double time{};
    double sample_mean{};
    double theoretical_mean{};
    double relative_mean_error{};
    double sample_variance{};
    double theoretical_variance{};
    double relative_variance_error{};
};

[[nodiscard]] SampleStatistics summarize(std::span<const double> samples);

[[nodiscard]] ConfidenceInterval confidence_interval_95(
    double estimate,
    double standard_error) noexcept;

}  // namespace gpu_lsm

