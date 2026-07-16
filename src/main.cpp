#include "gpu_lsm/benchmark.hpp"
#include "gpu_lsm/config.hpp"
#include "gpu_lsm/path_matrix.hpp"

#include <iostream>

int main() {
    const gpu_lsm::MarketParams market{};
    const gpu_lsm::SimulationConfig simulation{};
    const gpu_lsm::OptionParams option{};

    const gpu_lsm::PathMatrix storage(
        simulation.num_time_steps + 1,
        simulation.num_paths);

    std::cout << "GPU LSM XVA Phase 1 skeleton\n"
              << "spot=" << market.spot << ", strike=" << option.strike
              << ", paths=" << simulation.num_paths
              << ", time_steps=" << simulation.num_time_steps << '\n'
              << "planned path storage="
              << gpu_lsm::bytes_to_mebibytes(storage.bytes()) << " MiB\n";
}

