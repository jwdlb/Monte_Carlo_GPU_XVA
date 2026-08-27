#pragma once

/*
 * Purpose: Declares the CPU Monte Carlo pricing and full-path-generation API.
 * It combines market/option inputs with path storage, diagnostics, and timing
 * data. Implementations are intentionally staged as the CPU reference grows.
 */

#include "gpu_lsm/benchmark.hpp"
#include "gpu_lsm/config.hpp"
#include "gpu_lsm/path_matrix.hpp"

#include <cstddef>
#include <limits>

namespace gpu_lsm {

// Numerical-health information collected while paths are being generated.
struct PathDiagnostics {
    double minimum_finite_state{std::numeric_limits<double>::infinity()};
    std::size_t non_positive_state_count{};
    std::size_t non_finite_state_count{};
};

// Richer path-generation result for callers that need timings and diagnostics.
struct PathGenerationResult {
    PathMatrix paths;
    StageTimings timings;
    PathDiagnostics diagnostics;
};

// Generates exact-GBM paths only; a convenient wrapper around detailed output.
[[nodiscard]] PathMatrix generate_exact_gbm_paths(
    const MarketParams& market,
    double maturity,
    const SimulationConfig& simulation);

// Generates paths plus timings and checks for non-positive/non-finite states.
[[nodiscard]] PathGenerationResult generate_exact_gbm_paths_with_timings(
    const MarketParams& market,
    double maturity,
    const SimulationConfig& simulation);

}  // namespace gpu_lsm
