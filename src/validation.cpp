#include "gpu_lsm/validation.hpp"

#include "gpu_lsm/black_scholes.hpp"

namespace gpu_lsm {

ValidationReport validate_bermudan_result(
    const MarketParams& market, const BermudanPutParams& option,
    const BermudanLsmResult& lsm) {
    // More exercise rights cannot reduce a put's value. These two references
    // bracket the Bermudan contract, while 4 standard errors avoid rejecting a
    // valid Monte Carlo estimate just because of ordinary simulation noise.
    const double european = black_scholes_put(market, option.strike, option.maturity);
    const double american = crr_american_put(market, option.strike, option.maturity);
    constexpr double confidence_multiplier = 4.0;
    const bool above = lsm.price + confidence_multiplier * lsm.standard_error >= european;
    const bool below = lsm.price - confidence_multiplier * lsm.standard_error <= american;
    return {european, american, above, below, above && below};
}

}  // namespace gpu_lsm
