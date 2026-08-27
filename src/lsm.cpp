#include "gpu_lsm/lsm.hpp"

#include "gpu_lsm/path_matrix.hpp"
#include "gpu_lsm/statistics.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gpu_lsm {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}


// LSM needs to fit this polynomial:
//   continuation value = a0 + a1 * x + a3 * x^2
// where x = S(t) / K is the stock price normalised by the strike.

// a0, a1, a2 are the regression coefficients.
// a0 is the intercept, a1 is the linear term, and a2 is the quadratic term.
// That means it needs to solve three simultaneous equations. solve_3x3 does that.

// The input augmented is a 3-by-4 table:
// [a b c | d]
// [e f g | h]
// [i j k | l]
// The first three columns are the regression system; the final column is the right-hand side.
// The function uses Gaussian elimination:
// 1. Find the largest available pivot.
// 2. Swap it into the current row.
// 3. Divide the row so the pivot becomes 1.
// 4. Eliminate that column from the other rows.
// 5. Read the three answers from the last column.
bool solve_3x3(std::array<std::array<double, 4>, 3>& augmented, std::array<double, 3>& result) {
    // Solve [X'X | X'Y] for the three regression coefficients. Pivoting avoids
    // dividing by an unnecessarily small entry; near-singular systems fail.

    for (std::size_t column = 0; column < 3; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < 3; ++row) {
            if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column])) pivot = row;
        }
        if (std::abs(augmented[pivot][column]) < 1e-12) return false;
        std::swap(augmented[pivot], augmented[column]);
        const double divisor = augmented[column][column];
        for (std::size_t entry = column; entry < 4; ++entry) augmented[column][entry] /= divisor;
        for (std::size_t row = 0; row < 3; ++row) {
            if (row == column) continue;
            const double factor = augmented[row][column];
            for (std::size_t entry = column; entry < 4; ++entry) {
                augmented[row][entry] -= factor * augmented[column][entry];
            }
        }
    }
    for (std::size_t row = 0; row < 3; ++row) result[row] = augmented[row][3];
    return true;
}


}  // namespace

// Prices a Bermudan put option using the Longstaff-Schwartz Monte Carlo method. 

// Inputs:
// market      -  rate, volatility, etc.
// option      -  strike, maturity, exercise dates
// simulation  -  path count and time steps
// paths       -  all simulated stock prices from GBM

// The returned BermudanLsmResult includes:
// lsm.price               -  estimated option value today
// lsm.standard_error      -  uncertainty from random simulation
// lsm.exercise_indices    -  chosen exercise date for every path
// lsm.exercise_cashflows  -  payoff received on every path

BermudanLsmResult price_bermudan_put_lsm(const MarketParams& market, const BermudanPutParams& option,const SimulationConfig& simulation, const PathMatrix& paths) {
    // Checks that the model inputs are valid.
    validate(market);
    validate(simulation);
    validate(option, simulation);
    // Ensures the supplied path matrix has the expected dimensions.
    if (paths.num_times() != simulation.num_time_steps + 1
        || paths.num_paths() != simulation.num_paths) {
        throw std::invalid_argument("path matrix does not match simulation configuration");
    }

    // LSM works backwards: terminal payoffs are known, then earlier exercise
    // decisions compare exercising now with an estimate of waiting.
    const auto total_start = Clock::now();
    const double dt = option.maturity / static_cast<double>(simulation.num_time_steps);  // This calculates one time step: 1/52 years for weekly steps, 1/12 for monthly steps, etc.
    const std::size_t maturity_index = option.exercise_indices.back();  // Gets the final allowed exercise date. For the default contract: maturity_index = 52
    std::vector<std::size_t> exercise_indices(simulation.num_paths, maturity_index);  // Creates one exercise-date entry per path. Initially, all paths are assumed to exercise at maturity:
    std::vector<double> cashflows(simulation.num_paths);   // Creates one payoff entry per path. This will be updated as earlier exercise dates are considered. Initially, all paths are assumed to exercise at maturity, so the cashflows vector is filled with the terminal payoff at maturity.

    // Calculates the put payoff at maturity: max(K - S(T), 0) for every path. This is the last exercise date, so the payoff is known with certainty.
    // At this point, we are effectively treating the option as a European put. The backwards LSM loop will add the Bermudan early-exercise feature.
    
    // Start by assuming every path waits to maturity. Earlier dates may replace
    // the selected cashflow when immediate exercise is better.
    for (std::size_t path = 0; path < simulation.num_paths; ++path) {
        cashflows[path] = std::max(option.strike - paths(maturity_index, path), 0.0);
    }


    StageTimings timings{};
    // Skip maturity (already valued) and visit exercise dates backwards.
    // Start before maturity.    Go backwards through earlier exercise dates.    Stop before index 0 would underflow.
    for (std::size_t exercise_pos = option.exercise_indices.size() - 1; exercise_pos-- > 0;) {
        // Gets the actual grid date for this exercise opportunity. For the default contract, the first exercise date is index 13, which is 13 weeks after the start.
        const std::size_t time_index = option.exercise_indices[exercise_pos];
        // Calculates the time for this exercise opportunity.
        const double time = dt * static_cast<double>(time_index);
        const auto regression_start = Clock::now();

        // Creates an initially-zero 3-by-4 regression system and an in-the-money path counter.
        std::array<std::array<double, 4>, 3> normal_equations{};
        std::size_t itm_count = 0;


        // Fit continuation only on in-the-money paths: only they can sensibly
        // exercise a put at this date.

        // Looks at every path at the current exercise date.
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            // Calculates what exercising immediately would pay.
            const double payoff = std::max(option.strike - paths(time_index, path), 0.0);
            // If the option is out-of-the-money, it cannot be exercised, so skip this path.
            if (payoff <= 0.0) continue;

            // Creates normalised stock price: x = S(t) / K. This is the independent variable for the regression.
            const double x = paths(time_index, path) / option.strike;
            // Creates the polynomial regression inputs:
            const std::array<double, 3> basis{1.0, x, x * x};

            // cashflows[path] is the currently selected later payoff for that path.
            // exercise_indices[path] is when that later payoff happens.
            // The code discounts that later payoff back to the current decision date:
            // discounted cashflow = cashflow * exp(-r * dt * (exercise_index - time_index))
            const double response = cashflows[path] * std::exp(
                -market.rate * dt * static_cast<double>(exercise_indices[path] - time_index));

            // Builds the standard least-squares regression equations:
            // [X'X | X'Y] = sum(basis * basis' | basis * response) over all in-the-money paths.
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 3; ++column) {
                    normal_equations[row][column] += basis[row] * basis[column];
                }
                normal_equations[row][3] += basis[row] * response;
            }
            ++itm_count;  // Counts how many paths were in-the-money at this exercise date.
        }

        // Solves the regression system to get the continuation-value polynomial coefficients.
        std::array<double, 3> coefficients{};
        const bool fitted = itm_count >= 3 && solve_3x3(normal_equations, coefficients);   // At least three paths are required because there are three coefficients.
        timings.statistics_ms += elapsed_ms(regression_start, Clock::now());
        // If there is insufficient independent regression data, retain the
        // later exercise policy already established.
        if (!fitted) continue;

        const auto exercise_start = Clock::now();

        // Use the fitted polynomial to make the exercise-versus-wait decision.
        // Loops again over each path to make its exercise decision.
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            // Again, only in-the-money paths can exercise.
            const double payoff = std::max(option.strike - paths(time_index, path), 0.0);
            if (payoff <= 0.0) continue;

            // Calculates the normalised stock price for this path at this exercise date.
            const double x = paths(time_index, path) / option.strike;

            // Uses the fitted polynomial to estimate the value of waiting: C(t) = a0 + a1 * x + a2 * x^2. If the immediate payoff is better, exercise now.
            const double continuation = coefficients[0] + coefficients[1] * x + coefficients[2] * x * x;
            // This is the actual Bermudan exercise rule:
            // if exercising now is worth more than waiting:
            //     exercise now
            // otherwise:
            //     keep the later exercise decision
            if (payoff > continuation) {
                exercise_indices[path] = time_index;
                cashflows[path] = payoff;
            }
        }
        timings.terminal_payoff_ms += elapsed_ms(exercise_start, Clock::now());
        (void)time;
    }

    // Discount each chosen path cashflow to time zero; their average is the
    // Monte Carlo price and their dispersion produces its standard error.

    // After every path has a final exercise date and cashflow:

    // Creates storage for each path’s value today.
    std::vector<double> discounted(simulation.num_paths);

    // Discounts the path’s selected cashflow from its exercise date back to today. path’s value today = cashflow * exp(-r * dt * (exercise_index - 0))
    for (std::size_t path = 0; path < simulation.num_paths; ++path) {
        discounted[path] = cashflows[path] * std::exp(
            -market.rate * dt * static_cast<double>(exercise_indices[path]));
    }

    // Calculates: 
    // stats.mean            -  average discounted payoff = option price
    // stats.standard_error  -  simulation uncertainty
    const auto stats = summarize(discounted);

    // Builds a 95% confidence interval around the Monte Carlo estimate.
    const auto interval = confidence_interval_95(stats.mean, stats.standard_error);
    timings.total_ms = elapsed_ms(total_start, Clock::now());
    // Returns the final LSM result.
    return {stats.mean, stats.standard_error, interval.low, interval.high,
            std::move(exercise_indices), std::move(cashflows), timings};
}

}  // namespace gpu_lsm
