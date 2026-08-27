/*
 * Purpose: Implements the analytical probability helper and Black--Scholes
 * European-put benchmark used to validate Monte Carlo estimates.
 */

#include "gpu_lsm/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gpu_lsm {

// Express the normal CDF through erfc for stable standard-library evaluation.
double normal_cdf(double value) noexcept {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double black_scholes_put(
    const MarketParams& market,
    double strike,
    double maturity) {
    // Validate before logarithms, square roots, divisions, or exponentials.
    validate(market);
    const SimulationConfig grid{};
    validate(BermudanPutParams{strike, maturity, {grid.num_time_steps}}, grid);

    const double discounted_strike = strike * std::exp(-market.rate * maturity);
    const double discounted_spot =
        market.spot * std::exp(-market.dividend_yield * maturity);
    if (!std::isfinite(discounted_strike) || !std::isfinite(discounted_spot)) {
        throw std::overflow_error("Black-Scholes discount factor overflow");
    }

    // With no diffusion the terminal spot is deterministic, and the usual
    // d1/d2 expression would divide by zero.
    if (market.volatility == 0.0) {
        return std::max(discounted_strike - discounted_spot, 0.0);
    }

    const double sqrt_maturity = std::sqrt(maturity);
    const double volatility_squared = market.volatility * market.volatility;
    const double d1 =
        (std::log(market.spot / strike)
         + (market.rate - market.dividend_yield + 0.5 * volatility_squared)
               * maturity)
        / (market.volatility * sqrt_maturity);
    const double d2 = d1 - market.volatility * sqrt_maturity;
    const double price =
        discounted_strike * normal_cdf(-d2)
        - discounted_spot * normal_cdf(-d1);

    if (!std::isfinite(price)) {
        throw std::overflow_error("Black-Scholes price is not finite");
    }

    // The analytical value is non-negative; suppress only round-off-sized
    // negative values produced by subtraction in extreme out-of-the-money cases.
    return std::max(price, 0.0);
}

double crr_american_put(
    const MarketParams& market, double strike, double maturity, std::size_t steps) {
    // A recombining Cox-Ross-Rubinstein tree is used only as a high-accuracy
    // American-put validation reference; it is not part of the LSM workflow.
    validate(market);
    const SimulationConfig grid{};
    validate(BermudanPutParams{strike, maturity, {grid.num_time_steps}}, grid);
    if (steps == 0) throw std::invalid_argument("binomial steps must be positive");
    if (market.volatility == 0.0) {
        return std::max(strike - market.spot, 0.0);
    }
    const double dt = maturity / static_cast<double>(steps);
    const double up = std::exp(market.volatility * std::sqrt(dt));
    const double down = 1.0 / up;
    const double growth = std::exp((market.rate - market.dividend_yield) * dt);
    const double probability = (growth - down) / (up - down);
    if (probability < 0.0 || probability > 1.0 || !std::isfinite(probability)) {
        throw std::invalid_argument("invalid CRR probability");
    }
    const double discount = std::exp(-market.rate * dt);
    std::vector<double> values(steps + 1);
    // Set terminal intrinsic values, then roll the tree backwards. At each
    // node the holder takes the better of immediate exercise and continuation.
    for (std::size_t j = 0; j <= steps; ++j) {
        const double spot = market.spot * std::pow(
            up, static_cast<double>(2 * j) - static_cast<double>(steps));
        values[j] = std::max(strike - spot, 0.0);
    }
    for (std::size_t time = steps; time-- > 0;) {
        for (std::size_t j = 0; j <= time; ++j) {
            const double spot = market.spot * std::pow(
                up, static_cast<double>(2 * j) - static_cast<double>(time));
            const double continuation = discount * (
                probability * values[j + 1] + (1.0 - probability) * values[j]);
            values[j] = std::max(strike - spot, continuation);
        }
    }
    return values.front();
}

}  // namespace gpu_lsm
