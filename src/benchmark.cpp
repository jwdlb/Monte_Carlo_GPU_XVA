/*
 * Purpose: Implements small reporting utilities shared by the executable and
 * future benchmark code. No simulation or financial calculation happens here.
 */

#include "gpu_lsm/benchmark.hpp"

namespace gpu_lsm {

// Use a binary MiB, rather than a decimal MB, for memory allocation reporting.
double bytes_to_mebibytes(std::size_t bytes) noexcept {
    constexpr double bytes_per_mebibyte = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_mebibyte;
}

}  // namespace gpu_lsm
