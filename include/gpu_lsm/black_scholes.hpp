#pragma once

/*
 * Purpose: Declares analytical Black--Scholes helpers. The analytical European
 * put is the deterministic reference value against which Monte Carlo will be
 * validated once both routines are implemented.
 */

#include "gpu_lsm/config.hpp"

namespace gpu_lsm {

// Standard-normal cumulative probability: P(Z <= value), where Z ~ N(0, 1).
[[nodiscard]] double normal_cdf(double value) noexcept;

// Analytical price of a European put under the Black--Scholes assumptions.
[[nodiscard]] double black_scholes_put(
    const MarketParams& market,
    double strike,
    double maturity);

// CRR American-put value used only as an upper-bound validation reference.
[[nodiscard]] double crr_american_put(
    const MarketParams& market,
    double strike,
    double maturity,
    std::size_t steps = 2'000);

}  // namespace gpu_lsm
