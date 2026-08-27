#pragma once

#include "gpu_lsm/config.hpp"
#include "gpu_lsm/cva.hpp"
#include "gpu_lsm/lsm.hpp"

namespace gpu_lsm {

struct ValidationReport {
    // Analytical European lower reference and CRR American upper reference.
    double european_put{};
    double american_put{};
    bool above_european{};
    bool below_american{};
    bool passed{};
};

// Checks the LSM estimate against the no-arbitrage ordering European <=
// Bermudan <= American, with a tolerance based on its standard error.
[[nodiscard]] ValidationReport validate_bermudan_result(
    const MarketParams& market,
    const BermudanPutParams& option,
    const BermudanLsmResult& lsm);

}  // namespace gpu_lsm
