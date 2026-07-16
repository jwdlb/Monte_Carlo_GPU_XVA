#pragma once

#include "gpu_lsm/benchmark.hpp"
#include "gpu_lsm/config.hpp"
#include "gpu_lsm/evolution_scheme.hpp"
#include "gpu_lsm/path_matrix.hpp"

namespace gpu_lsm {

struct MonteCarloResult {
    double price{};
    double standard_error{};
    double confidence_low{};
    double confidence_high{};
    double runtime_ms{};
};

struct PathGenerationResult {
    PathMatrix paths;
    StageTimings timings;
};

[[nodiscard]] MonteCarloResult price_european_put_terminal_mc(
    const MarketParams& market,
    const OptionParams& option,
    const SimulationConfig& simulation);

[[nodiscard]] PathMatrix generate_paths(
    const MarketParams& market,
    double maturity,
    const SimulationConfig& simulation,
    EvolutionScheme scheme);

[[nodiscard]] PathGenerationResult generate_paths_with_timings(
    const MarketParams& market,
    double maturity,
    const SimulationConfig& simulation,
    EvolutionScheme scheme);

}  // namespace gpu_lsm

