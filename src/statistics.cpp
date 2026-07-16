#include "gpu_lsm/statistics.hpp"

#include <stdexcept>

namespace gpu_lsm {

SampleStatistics summarize(std::span<const double> samples) {
    if (samples.empty()) {
        throw std::invalid_argument("Cannot summarize an empty sample");
    }
    throw std::logic_error("summarize is not implemented yet");
}

ConfidenceInterval confidence_interval_95(
    double estimate,
    double standard_error) noexcept {
    constexpr double z_95 = 1.959963984540054;
    return {
        estimate - z_95 * standard_error,
        estimate + z_95 * standard_error,
    };
}

}  // namespace gpu_lsm

