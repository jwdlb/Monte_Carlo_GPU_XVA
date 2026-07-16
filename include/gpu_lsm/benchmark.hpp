#pragma once

#include <cstddef>

namespace gpu_lsm {

struct StageTimings {
    double random_generation_ms{};
    double path_evolution_ms{};
    double terminal_payoff_ms{};
    double statistics_ms{};
    double total_ms{};
};

struct MemoryReport {
    std::size_t path_matrix_bytes{};
};

[[nodiscard]] double bytes_to_mebibytes(std::size_t bytes) noexcept;

}  // namespace gpu_lsm

