#pragma once

#include "gpu_lsm/benchmark.hpp"
#include "gpu_lsm/config.hpp"

#include <cstddef>
#include <vector>

namespace gpu_lsm {

class PathMatrix;

struct BermudanLsmResult {
    // Time-zero risk-free value and its Monte Carlo sampling uncertainty.
    double price{};
    double standard_error{};
    double confidence_low{};
    double confidence_high{};
    // One selected exercise date/cashflow for each simulated path.  CVA uses
    // these pathwise results to determine when exposure remains outstanding.
    std::vector<std::size_t> exercise_indices;
    std::vector<double> exercise_cashflows;
    StageTimings timings{};
};

// Inputs: market/contract/simulation assumptions plus a matching price-path
// matrix. Output: price statistics and the pathwise exercise policy found by
// backward Longstaff--Schwartz continuation-value regression.
[[nodiscard]] BermudanLsmResult price_bermudan_put_lsm(
    const MarketParams& market,
    const BermudanPutParams& option,
    const SimulationConfig& simulation,
    const PathMatrix& paths);

}  // namespace gpu_lsm
