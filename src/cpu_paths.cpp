#include "gpu_lsm/cpu_paths.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

namespace gpu_lsm {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}  // namespace

void validate(const MarketParams& market) {
    if (!std::isfinite(market.spot) || !std::isfinite(market.rate)
        || !std::isfinite(market.dividend_yield) || !std::isfinite(market.volatility)) {
        throw std::invalid_argument("market inputs must be finite");
    }
    if (market.spot <= 0.0 || market.volatility < 0.0) {
        throw std::invalid_argument("spot must be positive and volatility non-negative");
    }
}

void validate(const SimulationConfig& simulation) {
    if (simulation.num_paths == 0 || simulation.num_time_steps == 0) {
        throw std::invalid_argument("path count and time-step count must be positive");
    }
}

void validate(const BermudanPutParams& option, const SimulationConfig& simulation) {
    if (!std::isfinite(option.strike) || !std::isfinite(option.maturity)
        || option.strike <= 0.0 || option.maturity <= 0.0) {
        throw std::invalid_argument("Bermudan strike and maturity must be finite and positive");
    }
    if (option.exercise_indices.empty()) {
        throw std::invalid_argument("Bermudan option needs at least one exercise date");
    }
    std::size_t previous = 0;
    for (const auto index : option.exercise_indices) {
        if (index == 0 || index > simulation.num_time_steps || index <= previous) {
            throw std::invalid_argument("exercise indices must be strictly increasing grid indices");
        }
        previous = index;
    }
    if (option.exercise_indices.back() != simulation.num_time_steps) {
        throw std::invalid_argument("Bermudan maturity must be an exercise date");
    }
}

void validate(const CreditParams& credit) {
    if (!std::isfinite(credit.hazard_rate) || !std::isfinite(credit.recovery_rate)
        || credit.hazard_rate < 0.0 || credit.recovery_rate < 0.0
        || credit.recovery_rate > 1.0) {
        throw std::invalid_argument("invalid credit parameters");
    }
}

PathMatrix generate_exact_gbm_paths(
    const MarketParams& market, double maturity, const SimulationConfig& simulation) {
    return generate_exact_gbm_paths_with_timings(market, maturity, simulation).paths;
}

PathGenerationResult generate_exact_gbm_paths_with_timings(const MarketParams& market, double maturity, const SimulationConfig& simulation) {
    // Valdate all inputs before allocating memory or starting the clock.
    validate(market);
    validate(simulation);
    if (!std::isfinite(maturity) || maturity <= 0.0) {
        throw std::invalid_argument("maturity must be finite and positive");
    }

    // Starts the clock and allocates the path matrix, initializing the first time slice to the spot price.
    const auto total_start = Clock::now();
    PathMatrix paths(simulation.num_time_steps + 1, simulation.num_paths);   // +1 for the initial time slice
    for (double& value : paths.time_slice(0)) value = market.spot;

    // Cleanly compute the GBM drift and diffusion coefficients for the time step.
    const double dt = maturity / static_cast<double>(simulation.num_time_steps);  // length of one simulation interval, defaults to 1/52 years
    
    // Together, they are used in:
    // S(t + dt) = S(t) * exp((r - q - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    const double drift = (market.rate - market.dividend_yield - 0.5 * market.volatility * market.volatility) * dt;   // This is the non-random component in the GBM exponential formula
    const double diffusion = market.volatility * std::sqrt(dt);  // This is the amount by which each random shock affects the price

    // setup the random number generator and normal distribution for the GBM shocks, and prepare diagnostics and timings.
    std::mt19937_64 engine(simulation.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    PathDiagnostics diagnostics{market.spot, 0, 0};
    StageTimings timings{};

    // Evolve the paths through time, generating a new random shock for each path at each time step. start at one as first price is pre filled
    for (std::size_t time = 1; time <= simulation.num_time_steps; ++time) {
        const auto rng_start = Clock::now();
        // current is the full row of prices for the current time point. For now, this storage will temporarily hold random shocks.
        auto current = paths.time_slice(time);
        // For every path, generate one standard-normal random number. using refeanced current to do this in place, so that we don't need to allocate a separate vector for the random numbers.
        for (double& value : current) value = normal(engine);

        // Adds the time spent generating random numbers to the timing report.
        timings.random_generation_ms += elapsed_ms(rng_start, Clock::now());

        // Evolve every path to the next time step using the GBM formula, and update diagnostics.
        const auto evolve_start = Clock::now();
        // previous is the row containing prices from the prior week.
        const auto previous = paths.time_slice(time - 1);

        // Loops through every individual Monte Carlo scenario.
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            // GBM formula:
            //  previous[path]  -  previous stock price S(t)
            //  current[path]   -  random normal shock Z
            //  state           -  next stock price S(t + dt)
            const double state = previous[path] * std::exp(drift + diffusion * current[path]);

            // Checks that the generated price is:
            // - finite: not NaN, not positive infinity, not negative infinity
            // - positive: stock prices should stay above zero under GBM
            // If invalid, the calculation immediately stops rather than returning unreliable results.
            if (!std::isfinite(state) || state <= 0.0) {
                ++diagnostics.non_finite_state_count;
                throw std::runtime_error("exact GBM produced invalid state");
            }

            // Overwrites the temporary random shock with the actual newly simulated stock price.
            current[path] = state;

            // Keeps track of the lowest valid stock price generated across every path and every date.
            diagnostics.minimum_finite_state = std::min(diagnostics.minimum_finite_state, state);
        }
        // Adds the time spent applying the GBM formula to all paths.
        timings.path_evolution_ms += elapsed_ms(evolve_start, Clock::now());
    }
    // Records the total runtime, including allocation, random-number generation, GBM evolution, and checks.
    timings.total_ms = elapsed_ms(total_start, Clock::now());

    // Creates and returns a PathGenerationResult.
    // std::move(paths) transfers ownership of the large PathMatrix into the result instead of copying millions of doubles. That is important for performance.
    return {std::move(paths), timings, diagnostics};
}

}  // namespace gpu_lsm
