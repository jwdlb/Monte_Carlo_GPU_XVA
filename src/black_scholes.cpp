#include "gpu_lsm/black_scholes.hpp"

#include <cmath>
#include <stdexcept>

namespace gpu_lsm {

double normal_cdf(double value) noexcept {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double black_scholes_put(
    const MarketParams& market,
    double strike,
    double maturity) {
    validate(market);
    validate(OptionParams{strike, maturity});
    throw std::logic_error("black_scholes_put is not implemented yet");
}

}  // namespace gpu_lsm

