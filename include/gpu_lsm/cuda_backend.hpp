#pragma once

#include "gpu_lsm/cva.hpp"
#include "gpu_lsm/lsm.hpp"

#include <cstddef>
#include <string>

namespace gpu_lsm {

// Summary returned by the optional CUDA implementation.  Full price paths
// remain on the device; only the existing pricing/CVA results are copied back.
struct CudaBermudanResult {
    BermudanLsmResult lsm;
    CvaResult cva;
    StageTimings path_timings;
    std::size_t device_path_bytes{};
    std::string device_name;
};

// True only when this build includes CUDA and a supported device is usable.
[[nodiscard]] bool cuda_backend_available() noexcept;
[[nodiscard]] std::string cuda_backend_description();

// Runs exact-GBM simulation, LSM and CVA without copying the path grid to host.
[[nodiscard]] CudaBermudanResult run_cuda_bermudan_xva(
    const MarketParams& market,
    const BermudanPutParams& option,
    const SimulationConfig& simulation,
    const CreditParams& credit);

}  // namespace gpu_lsm
