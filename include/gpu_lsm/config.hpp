#pragma once

#include <cstddef>
#include <cstdint>

namespace gpu_lsm {

struct MarketParams {
    double spot{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double volatility{0.20};
};

struct SimulationConfig {
    std::size_t num_paths{100'000};
    std::size_t num_time_steps{50};
    std::uint64_t seed{42};
};

struct OptionParams {
    double strike{100.0};
    double maturity{1.0};
};

// Throws std::invalid_argument when Phase 1 input requirements are violated.
void validate(const MarketParams& market);
void validate(const SimulationConfig& simulation);
void validate(const OptionParams& option);

}  // namespace gpu_lsm

