#include "gpu_lsm/cpu_paths.hpp"
#include "gpu_lsm/cuda_backend.hpp"
#include "gpu_lsm/cva.hpp"
#include "gpu_lsm/lsm.hpp"
#include "gpu_lsm/validation.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
// Runs the full pricing pipeline once. `main` calls this either for the normal
// validation case or for each requested benchmark size.
int run_case(std::size_t paths) {
    // `{}` selects the defaults declared in config.hpp; together, these are
    // the market, contract, credit, and Monte Carlo inputs.
    gpu_lsm::MarketParams market{};
    gpu_lsm::BermudanPutParams option{};
    gpu_lsm::CreditParams credit{};
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = paths;
    if (gpu_lsm::cuda_backend_available()) {
        const auto gpu = gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit);
        const auto validation = gpu_lsm::validate_bermudan_result(market, option, gpu.lsm);
        std::cout << "backend=cuda device=" << gpu.device_name << " paths=" << paths
                  << " price=" << gpu.lsm.price << " se=" << gpu.lsm.standard_error
                  << " cva=" << gpu.cva.cva
                  << " memory_mib=" << gpu_lsm::bytes_to_mebibytes(gpu.device_path_bytes)
                  << " workspace_mib=" << gpu_lsm::bytes_to_mebibytes(gpu.device_workspace_bytes)
                  << " path_ms=" << gpu.path_timings.total_ms
                  << " lsm_ms=" << gpu.lsm.timings.total_ms
                  << " cva_ms=" << gpu.cva.timings.total_ms << '\n';
        if (paths == 100'000) {
            std::cout << "European=" << validation.european_put
                      << " American=" << validation.american_put
                      << " validation=" << (validation.passed ? "PASS" : "FAIL") << '\n';
        }
        return validation.passed ? 0 : 1;
    }
    // Stage 1: simulate stock-price paths. The result owns the paths and also
    // contains performance timings and numerical-health diagnostics.
    const gpu_lsm::PathGenerationResult generated = gpu_lsm::generate_exact_gbm_paths_with_timings(
        market, option.maturity, simulation);
    // Stage 2: choose the best exercise date per path and estimate the
    // risk-free Bermudan option price from the simulated cashflows.
    const gpu_lsm::BermudanLsmResult lsm = gpu_lsm::price_bermudan_put_lsm(
        market, option, simulation, generated.paths);
    // Stage 3: aggregate the remaining LSM cashflows into exposure and then
    // calculate the expected loss from counterparty default (CVA).
    const gpu_lsm::CvaResult cva = gpu_lsm::calculate_unilateral_cva(
        market, option, simulation, credit, lsm);
    // A Bermudan put should fall between European and American put reference
    // values, allowing for Monte Carlo sampling uncertainty.
    const gpu_lsm::ValidationReport validation = gpu_lsm::validate_bermudan_result(
        market, option, lsm);
    std::cout << "paths=" << paths << " price=" << lsm.price
              << " se=" << lsm.standard_error << " cva=" << cva.cva
              << " memory_mib=" << gpu_lsm::bytes_to_mebibytes(generated.paths.bytes())
              << " path_ms=" << generated.timings.total_ms
              << " lsm_ms=" << lsm.timings.total_ms << " cva_ms=" << cva.timings.total_ms
              << " total_ms=" << (generated.timings.total_ms + lsm.timings.total_ms + cva.timings.total_ms)
              << '\n';
    // The deterministic reference prices do not change with path count, so
    // print them once rather than in every benchmark row.
    if (paths == 100'000) {
        std::cout << "European=" << validation.european_put
                  << " American=" << validation.american_put
                  << " validation=" << (validation.passed ? "PASS" : "FAIL") << '\n';
    }
    return validation.passed ? 0 : 1;
}

double relative_error_percent(double candidate, double baseline) {
    return baseline == 0.0 ? 0.0 : 100.0 * std::abs(candidate - baseline) / std::abs(baseline);
}

int run_compression_benchmark(std::size_t paths) {
    if (!gpu_lsm::cuda_backend_available()) {
        std::cerr << "compression-benchmark requires a compatible CUDA device\n";
        return 1;
    }
    const gpu_lsm::MarketParams market{};
    const gpu_lsm::BermudanPutParams option{};
    const gpu_lsm::CreditParams credit{};
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = paths;

    struct Variant {
        const char* name;
        gpu_lsm::CudaPipelineOptions options;
    };
    const std::array variants{
        Variant{"baseline-double-full-cva", {}},
        Variant{"exercise-date-cva", {.cva_exercise_dates_only = true}},
        Variant{"float-paths", {.float_paths = true}},
        Variant{"float-paths-and-exercise-date-cva",
                {.cva_exercise_dates_only = true, .float_paths = true}}
    };

    // One warm-up removes most CUDA context/JIT initialization from the comparison.
    (void)gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit);
    std::array<gpu_lsm::CudaBermudanResult, variants.size()> results;
    for (std::size_t i = 0; i < variants.size(); ++i)
        results[i] = gpu_lsm::run_cuda_bermudan_xva(
            market, option, simulation, credit, variants[i].options);

    const auto& baseline = results[0];
    const double baseline_total = baseline.path_timings.total_ms
        + baseline.lsm.timings.total_ms + baseline.cva.timings.total_ms;
    std::cout << std::fixed << std::setprecision(6)
              << "device: " << baseline.device_name << "\n"
              << "paths: " << paths << "\n\n"
              << "| variant | price | price error % | CVA | CVA error % | path MiB | workspace MiB | exposure dates | path ms | LSM ms | CVA ms | total ms | speedup |\n"
              << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (std::size_t i = 0; i < variants.size(); ++i) {
        const auto& result = results[i];
        const double total = result.path_timings.total_ms
            + result.lsm.timings.total_ms + result.cva.timings.total_ms;
        std::cout << "| " << variants[i].name
                  << " | " << result.lsm.price
                  << " | " << relative_error_percent(result.lsm.price, baseline.lsm.price)
                  << " | " << result.cva.cva
                  << " | " << relative_error_percent(result.cva.cva, baseline.cva.cva)
                  << " | " << gpu_lsm::bytes_to_mebibytes(result.device_path_bytes)
                  << " | " << gpu_lsm::bytes_to_mebibytes(result.device_workspace_bytes)
                  << " | " << result.cva.exposure.times.size()
                  << " | " << result.path_timings.total_ms
                  << " | " << result.lsm.timings.total_ms
                  << " | " << result.cva.timings.total_ms
                  << " | " << total
                  << " | " << (total == 0.0 ? 0.0 : baseline_total / total)
                  << "x |\n";
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    // Default to one 100k-path validation run; `benchmark` runs three sizes.
    const std::string_view command = argc > 1 ? argv[1] : "validate";
    if (command == "validate") return run_case(100'000);
    if (command == "benchmark") {
        int status = 0;
        for (const std::size_t paths : {100'000U, 500'000U, 1'000'000U}) status |= run_case(paths);
        return status;
    }
    if (command == "compression-benchmark") {
        try {
            const std::size_t paths = argc > 2 ? std::stoull(argv[2]) : 100'000U;
            return run_compression_benchmark(paths);
        } catch (const std::exception& error) {
            std::cerr << "invalid path count: " << error.what() << '\n';
            return 2;
        }
    }
    std::cerr << "Usage: gpu_lsm_xva [validate|benchmark|compression-benchmark [paths]]\n";
    return 2;
}
