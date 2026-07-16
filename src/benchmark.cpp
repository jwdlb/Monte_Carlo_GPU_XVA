#include "gpu_lsm/benchmark.hpp"

namespace gpu_lsm {

double bytes_to_mebibytes(std::size_t bytes) noexcept {
    constexpr double bytes_per_mebibyte = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_mebibyte;
}

}  // namespace gpu_lsm

