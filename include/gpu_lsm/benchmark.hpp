#pragma once

/*
 * Purpose: Defines lightweight performance and memory-reporting types. These
 * let later CPU and GPU implementations report comparable timings by stage.
 */

#include <cstddef>

namespace gpu_lsm {

// Durations, in milliseconds, for the major stages of a pricing run.
struct StageTimings {
    double random_generation_ms{};
    double path_evolution_ms{};
    double terminal_payoff_ms{};
    double statistics_ms{};
    double total_ms{};
};

// Memory consumption relevant to the full simulated path store.
struct MemoryReport {
    std::size_t path_matrix_bytes{};
};

// Converts bytes to binary mebibytes (1 MiB = 1024 * 1024 bytes).
[[nodiscard]] double bytes_to_mebibytes(std::size_t bytes) noexcept;

}  // namespace gpu_lsm
