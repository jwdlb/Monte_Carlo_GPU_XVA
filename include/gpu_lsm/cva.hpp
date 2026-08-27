#pragma once

#include "gpu_lsm/benchmark.hpp"
#include "gpu_lsm/config.hpp"
#include "gpu_lsm/lsm.hpp"

#include <vector>

namespace gpu_lsm {

struct ExposureProfile {
    // Parallel vectors: expected_exposure[i] is the average outstanding value
    // across all paths at times[i].
    std::vector<double> times;
    std::vector<double> expected_exposure;
};

struct CvaResult {
    // Expected loss from counterparty default and the profile used to form it.
    double cva{};
    ExposureProfile exposure;
    StageTimings timings{};
};

// Calculates unilateral CVA under independent, flat default intensity. Inputs
// include the LSM exercise policy; no new price-path simulation is performed.
[[nodiscard]] CvaResult calculate_unilateral_cva(
    const MarketParams& market,
    const BermudanPutParams& option,
    const SimulationConfig& simulation,
    const CreditParams& credit,
    const BermudanLsmResult& lsm);

}  // namespace gpu_lsm
