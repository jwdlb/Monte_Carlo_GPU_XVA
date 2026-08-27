#include "gpu_lsm/cva.hpp"

#include "gpu_lsm/statistics.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace gpu_lsm {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}

// calculates the expected credit loss caused by the counterparty potentially defaulting while the option still has value.

// Its inputs are:
// market       -  interest rate, etc.
// option       -  strike, maturity, exercise dates
// simulation   -  number of paths and time steps
// credit       -  hazard rate and recovery rate
// lsm          -  LSM price plus path-by-path exercise decisions

// Its output is:
// CvaResult {
//     cva,       // one final expected-loss number
//     exposure,  // expected exposure at every time point
//     timings
// }

CvaResult calculate_unilateral_cva(const MarketParams& market, const BermudanPutParams& option, const SimulationConfig& simulation, const CreditParams& credit, const BermudanLsmResult& lsm) {
    // Checks that all inputs make sense.
    validate(market);
    validate(simulation);
    validate(option, simulation);
    validate(credit);

    // LSM must provide one exercise decision and one cashflow for each Monte Carlo path.
    if (lsm.exercise_indices.size() != simulation.num_paths
        || lsm.exercise_cashflows.size() != simulation.num_paths) {
        throw std::invalid_argument("LSM policy does not match path count");
    }

    // CVA reuses LSM’s policy; it does not generate a new set of market paths.
    const auto start = Clock::now();
    const double dt = option.maturity / static_cast<double>(simulation.num_time_steps);  // Calculates the length of one time step.

    // Creates an empty exposure-profile object.
    ExposureProfile profile;

    // Creates two vectors with one entry per simulation date.
    // profile.times:
    // [0 years, week 1, week 2, ..., week 52]

    // profile.expected_exposure:
    // [EE at today, EE at week 1, ..., EE at week 52]
    profile.times.resize(simulation.num_time_steps + 1);
    profile.expected_exposure.resize(simulation.num_time_steps + 1);

    // Average the still-outstanding option value at every date. A path has no
    // exposure after it has exercised and received its selected cashflow.

    // Loop through every simulation date:
    for (std::size_t time_index = 0; time_index <= simulation.num_time_steps; ++time_index) {
        // Turns the grid index into an actual year fraction. For example:
        // time_index = 0  -> 0 years
        // time_index = 26 -> 0.5 years
        // time_index = 52 -> 1.0 year
        const double time = dt * static_cast<double>(time_index);
        profile.times[time_index] = time;

        // This will accumulate exposure across all paths at this one date.
        double total = 0.0;
        // Loop through every simulated path.
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            if (time_index <= lsm.exercise_indices[path]) {  // Ask whether that option position is still alive at this date. After exercise/payment, the contract is assumed to be settled, so there is no remaining counterparty exposure.
                // If the contract is still alive, calculate the future selected cashflow’s value at the current date. Value today = cashflow * exp(-r * dt * (exercise_index - current_index)). This is the discounted value of the cashflow that will be received at the path’s selected exercise date.
                total += lsm.exercise_cashflows[path] * std::exp(-market.rate * (dt * static_cast<double>(lsm.exercise_indices[path] - time_index)));
            }
        }
        // Average the exposure across every path: Expected Exposure (EE(time)) = sum(all outstanding path values) / number of paths. This is the expected exposure at this one date.
        profile.expected_exposure[time_index] = total / static_cast<double>(simulation.num_paths);
    }
    
    // Add expected loss interval by interval: discounted exposure × default
    // probability for the interval × loss-given-default (1 - recovery).
    double cva = 0.0;  // Start the final expected-loss total at zero.
    // Loop through every interval between two consecutive simulation dates.
    for (std::size_t index = 1; index < profile.times.size(); ++index) {
        // Get the start and end of the current interval.
        const double time = profile.times[index];
        const double previous = profile.times[index - 1];

        // This calculates the probability of default in that interval under a constant hazard-rate model.
        // The survival probability at time t is exp(-hazard_rate * t). The probability of default in the interval (previous, time] is the difference between the survival probabilities at the start and end of the interval.
        // Probability alive at start - probability alive at end = probability defaulted during that interval
        const double default_increment = std::exp(-credit.hazard_rate * previous) - std::exp(-credit.hazard_rate * time);

        // Calculate the expected loss for this interval: discounted exposure × default probability × loss-given-default.

        // This adds expected credit loss for the current interval: CVA increments = D(0, t) × EE(t) × PD(t - dt, t) × LGD. D(0, t) is the discount factor from today to time t. EE(t) is the expected exposure at time t. PD(t - dt, t) is the probability of default in the interval (t - dt, t]. LGD = 1 - recovery rate.
        cva += std::exp(-market.rate * time) * profile.expected_exposure[index]
            * default_increment * (1.0 - credit.recovery_rate);
    }
    // returns:
    // cva                 -  final expected counterparty-loss amount
    // profile             -  time grid plus expected exposure values
    // timings             -  total time spent
    StageTimings timings{};
    timings.total_ms = elapsed_ms(start, Clock::now());
    return {cva, std::move(profile), timings};
}

}  // namespace gpu_lsm
