#include "gpu_lsm/cpu_paths.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace gpu_lsm {
namespace {

void require_finite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string{name} + " must be finite");
    }
}

}  // namespace

void validate(const MarketParams& market) {
    require_finite(market.spot, "spot");
    require_finite(market.rate, "rate");
    require_finite(market.dividend_yield, "dividend yield");
    require_finite(market.volatility, "volatility");
    if (market.spot <= 0.0) {
        throw std::invalid_argument("spot must be positive");
    }
    if (market.volatility < 0.0) {
        throw std::invalid_argument("volatility cannot be negative");
    }
}

void validate(const SimulationConfig& simulation) {
    if (simulation.num_paths == 0) {
        throw std::invalid_argument("path count must be positive");
    }
    if (simulation.num_time_steps == 0) {
        throw std::invalid_argument("time-step count must be positive");
    }
}

void validate(const OptionParams& option) {
    require_finite(option.strike, "strike");
    require_finite(option.maturity, "maturity");
    if (option.strike <= 0.0) {
        throw std::invalid_argument("strike must be positive");
    }
    if (option.maturity <= 0.0) {
        throw std::invalid_argument("maturity must be positive");
    }
}

MonteCarloResult price_european_put_terminal_mc(
    const MarketParams& market,
    const OptionParams& option,
    const SimulationConfig& simulation) {
    validate(market);
    validate(option);
    validate(simulation);
    throw std::logic_error("terminal-only Monte Carlo is not implemented yet");
}

PathMatrix generate_paths(
    const MarketParams& market,
    double maturity,
    const SimulationConfig& simulation,
    EvolutionScheme scheme) {
    return generate_paths_with_timings(market, maturity, simulation, scheme).paths;
}

PathGenerationResult generate_paths_with_timings(
    const MarketParams& market,
    double maturity,
    const SimulationConfig& simulation,
    EvolutionScheme /*scheme*/) {
    validate(market);
    validate(OptionParams{1.0, maturity});
    validate(simulation);
    throw std::logic_error("full-path generation is not implemented yet");
}

}  // namespace gpu_lsm

