#include "gpu_lsm/cuda_backend.hpp"
#include "gpu_lsm/validation.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

namespace {

void require_cuda_or_skip() {
    if (!gpu_lsm::cuda_backend_available()) SKIP("CUDA device is not available");
}

}  // namespace

TEST_CASE("optimized CUDA pipeline is repeatable and financially valid") {
    require_cuda_or_skip();
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 20'000;
    const gpu_lsm::MarketParams market{};
    const gpu_lsm::BermudanPutParams option{};
    const gpu_lsm::CreditParams credit{};

    const auto first = gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit);
    const auto second = gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit);

    REQUIRE(std::isfinite(first.lsm.price));
    REQUIRE(std::isfinite(first.lsm.standard_error));
    REQUIRE(std::isfinite(first.cva.cva));
    REQUIRE(first.lsm.price >= 0.0);
    REQUIRE(first.lsm.standard_error >= 0.0);
    REQUIRE(first.cva.cva >= 0.0);
    REQUIRE(first.device_workspace_bytes >= first.device_path_bytes);
    REQUIRE(first.cva.exposure.expected_exposure.size() == simulation.num_time_steps + 1);
    REQUIRE(std::all_of(first.cva.exposure.expected_exposure.begin(),
                        first.cva.exposure.expected_exposure.end(),
                        [](double exposure) { return std::isfinite(exposure) && exposure >= 0.0; }));
    REQUIRE(std::all_of(first.lsm.exercise_indices.begin(), first.lsm.exercise_indices.end(),
                        [&option](std::size_t index) {
                            return std::find(option.exercise_indices.begin(), option.exercise_indices.end(), index)
                                != option.exercise_indices.end();
                        }));
    REQUIRE(gpu_lsm::validate_bermudan_result(market, option, first.lsm).passed);

    // Philox streams are deterministic.  The block partials still reach global
    // sums through atomics, so compare floating results at round-off tolerance.
    REQUIRE(second.lsm.price == Catch::Approx(first.lsm.price).epsilon(1e-12));
    REQUIRE(second.cva.cva == Catch::Approx(first.cva.cva).epsilon(1e-12));
}

TEST_CASE("optimized CUDA CVA respects zero-risk limits") {
    require_cuda_or_skip();
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 10'000;
    const gpu_lsm::MarketParams market{};
    const gpu_lsm::BermudanPutParams option{};

    auto credit = gpu_lsm::CreditParams{};
    credit.hazard_rate = 0.0;
    REQUIRE(gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit).cva.cva
            == Catch::Approx(0.0));

    credit = gpu_lsm::CreditParams{};
    credit.recovery_rate = 1.0;
    REQUIRE(gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit).cva.cva
            == Catch::Approx(0.0));
}

TEST_CASE("exercise-date CVA compression reports its accuracy and work reduction") {
    require_cuda_or_skip();
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 50'000;
    const gpu_lsm::MarketParams market{};
    const gpu_lsm::BermudanPutParams option{};
    const gpu_lsm::CreditParams credit{};

    const auto baseline = gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit);
    gpu_lsm::CudaPipelineOptions compressed_options{};
    compressed_options.cva_exercise_dates_only = true;
    const auto compressed = gpu_lsm::run_cuda_bermudan_xva(
        market, option, simulation, credit, compressed_options);

    CAPTURE(baseline.cva.cva, compressed.cva.cva,
            baseline.cva.timings.total_ms, compressed.cva.timings.total_ms);
    REQUIRE(compressed.cva.exposure.expected_exposure.size() == option.exercise_indices.size());
    REQUIRE(compressed.cva.exposure.expected_exposure.size()
            < baseline.cva.exposure.expected_exposure.size());
    REQUIRE(compressed.lsm.price == Catch::Approx(baseline.lsm.price).epsilon(1e-12));
    // Quarterly endpoint integration is deliberately approximate; keep its
    // error visible while rejecting a financially material regression.
    REQUIRE(compressed.cva.cva == Catch::Approx(baseline.cva.cva).epsilon(0.10));
    REQUIRE(std::isfinite(compressed.cva.timings.total_ms));
}

TEST_CASE("float path compression halves path memory with controlled pricing error") {
    require_cuda_or_skip();
    gpu_lsm::SimulationConfig simulation{};
    simulation.num_paths = 50'000;
    const gpu_lsm::MarketParams market{};
    const gpu_lsm::BermudanPutParams option{};
    const gpu_lsm::CreditParams credit{};

    const auto baseline = gpu_lsm::run_cuda_bermudan_xva(market, option, simulation, credit);
    gpu_lsm::CudaPipelineOptions compressed_options{};
    compressed_options.float_paths = true;
    const auto compressed = gpu_lsm::run_cuda_bermudan_xva(
        market, option, simulation, credit, compressed_options);

    CAPTURE(baseline.lsm.price, compressed.lsm.price, baseline.cva.cva, compressed.cva.cva,
            baseline.path_timings.total_ms, compressed.path_timings.total_ms,
            baseline.lsm.timings.total_ms, compressed.lsm.timings.total_ms);
    REQUIRE(compressed.device_path_bytes * 2 == baseline.device_path_bytes);
    REQUIRE(compressed.device_workspace_bytes < baseline.device_workspace_bytes);
    REQUIRE(compressed.lsm.price == Catch::Approx(baseline.lsm.price).epsilon(0.01));
    REQUIRE(compressed.cva.cva == Catch::Approx(baseline.cva.cva).epsilon(0.01));
    REQUIRE(std::isfinite(compressed.path_timings.total_ms));
    REQUIRE(std::isfinite(compressed.lsm.timings.total_ms));
}
