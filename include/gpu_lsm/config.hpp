#pragma once

/*
 * Purpose: Defines the input data shared by pricing and simulation code.
 * These structs contain no pricing logic; validate(...) checks that callers
 * supplied values which are meaningful for the Phase 1 pricing model.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_lsm {

// Market quantities used by the risk-neutral share-price model.
struct MarketParams {
    double spot{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double volatility{0.20};
};

// Controls the size and reproducibility of a Monte Carlo run.
struct SimulationConfig {
    std::size_t num_paths{100'000};
    std::size_t num_time_steps{52};
    std::uint64_t seed{42};
};

// Contract terms and permitted exercise dates for the one-asset Bermudan put.
struct BermudanPutParams {
    double strike{100.0};
    double maturity{1.0};
    std::vector<std::size_t> exercise_indices{13, 26, 39, 52};
};

// Flat, independent counterparty-credit inputs for unilateral CVA.
struct CreditParams {
    double hazard_rate{0.02};
    double recovery_rate{0.40};
};

// Throw std::invalid_argument when Phase 1 input requirements are violated.
// References avoid copying the small input structs and const prevents mutation.
void validate(const MarketParams& market);
void validate(const SimulationConfig& simulation);
void validate(const BermudanPutParams& option, const SimulationConfig& simulation);
void validate(const CreditParams& credit);

}  // namespace gpu_lsm
