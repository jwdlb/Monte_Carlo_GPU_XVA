#include "gpu_lsm/cuda_backend.hpp"

#include <stdexcept>

namespace gpu_lsm {

bool cuda_backend_available() noexcept { return false; }

std::string cuda_backend_description() { return "CUDA backend not built"; }

CudaBermudanResult run_cuda_bermudan_xva(
    const MarketParams&, const BermudanPutParams&, const SimulationConfig&, const CreditParams&) {
    throw std::runtime_error("CUDA backend is not available in this build");
}

CudaBermudanResult run_cuda_bermudan_xva(
    const MarketParams&, const BermudanPutParams&, const SimulationConfig&, const CreditParams&,
    const CudaPipelineOptions&) {
    throw std::runtime_error("CUDA backend is not available in this build");
}

}  // namespace gpu_lsm
