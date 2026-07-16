#pragma once

#include "gpu_lsm/config.hpp"

namespace gpu_lsm {

[[nodiscard]] double normal_cdf(double value) noexcept;

[[nodiscard]] double black_scholes_put(
    const MarketParams& market,
    double strike,
    double maturity);

}  // namespace gpu_lsm

