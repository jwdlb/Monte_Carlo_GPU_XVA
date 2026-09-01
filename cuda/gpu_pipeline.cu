// GPU implementation of the full Bermudan-put pricing and CVA workflow.
//
// The three main stages are: GBM path generation, Longstaff-Schwartz exercise
// regression, and exposure/CVA calculation.  Keeping them together makes the
// device-resident data flow explicit and avoids unnecessary host transfers.
#include "gpu_lsm/cuda_backend.hpp"

#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpu_lsm {
namespace {

// =============================================================================
// CUDA resource and error-handling helpers
// =============================================================================

// Turn CUDA API failures into C++ exceptions with the operation and CUDA error.
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// RAII wrapper for one device allocation.  The buffer is freed automatically
// when it leaves scope, including when a CUDA operation throws an exception.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        // Allocate enough bytes for count values of T on the GPU.
        check_cuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }  // cudaFree is safe during cleanup.
    // Copying would duplicate the owning pointer, so forbid it.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    // Expose the raw device pointer and allocation size without transferring ownership.
    [[nodiscard]] T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }
private:
    T* data_{};
    std::size_t count_{};
};

// =============================================================================
// Stage 1: exact geometric-Brownian-motion path generation
// =============================================================================

// Generate one complete path per thread.  Keeping the Philox state and shock in
// registers avoids materialising and rereading a full device normal buffer.
template <typename PathValue>
__global__ void generate_gbm_paths(PathValue* __restrict__ paths, std::size_t paths_count,
                                   std::size_t time_steps, std::uint64_t seed, double spot,
                                   double drift, double diffusion) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths_count) return;

    curandStatePhilox4_32_10_t state;
    curand_init(static_cast<unsigned long long>(seed),
                static_cast<unsigned long long>(path), 0ULL, &state);
    double current = spot;
    paths[path] = static_cast<PathValue>(current);
    for (std::size_t time = 1; time <= time_steps; ++time) {
        current *= exp(drift + diffusion * curand_normal_double(&state));
        paths[time * paths_count + path] = static_cast<PathValue>(current);
    }
}

// At maturity a Bermudan put is settled for max(K - S(T), 0) on each path.
template <typename PathValue>
__global__ void initialise_terminal_payoffs(const PathValue* terminal, double* cashflows,
                                             std::size_t* exercise_indices, std::size_t paths_count,
                                             std::size_t maturity_index, double strike) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path < paths_count) {
        cashflows[path] = fmax(strike - static_cast<double>(terminal[path]), 0.0);  // Intrinsic put payoff.
        exercise_indices[path] = maturity_index;               // Initial stopping date is maturity.
    }
}

// =============================================================================
// Stage 2: Longstaff-Schwartz regression and early-exercise policy
// =============================================================================

// sums stores six symmetric X'X terms, three X'Y terms, then the ITM count.
template <typename PathValue>
__global__ void accumulate_regression(const PathValue* spot_row, const double* cashflows,
                                      const std::size_t* exercise_indices, double* sums,
                                      std::size_t paths_count, std::size_t time_index,
                                      double strike, double rate_dt) {
    constexpr int component_count = 10;
    constexpr int warp_count = 8;  // The pipeline launches 256-thread blocks.
    __shared__ double warp_sums[component_count][warp_count];

    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double local[component_count]{};
    if (path < paths_count) {
        const double spot = static_cast<double>(spot_row[path]);
        const double payoff = fmax(strike - spot, 0.0);
        if (payoff > 0.0) {
            const double x = spot / strike;
            const double x2 = x * x;
            const double response = cashflows[path] * exp(-rate_dt *
                static_cast<double>(exercise_indices[path] - time_index));
            local[0] = 1.0; local[1] = x; local[2] = x2;
            local[3] = x2; local[4] = x * x2; local[5] = x2 * x2;
            local[6] = response; local[7] = x * response;
            local[8] = x2 * response; local[9] = 1.0;
        }
    }

    // All threads, including padding and out-of-the-money paths, participate
    // so a full-warp mask is valid and zero contributions remain harmless.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int component = 0; component < component_count; ++component)
            local[component] += __shfl_down_sync(0xffffffffU, local[component], offset);
    }

    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    if (lane == 0) {
#pragma unroll
        for (int component = 0; component < component_count; ++component)
            warp_sums[component][warp] = local[component];
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int component = 0; component < component_count; ++component)
            local[component] = lane < warp_count ? warp_sums[component][lane] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
            for (int component = 0; component < component_count; ++component)
                local[component] += __shfl_down_sync(0xffffffffU, local[component], offset);
        }
        if (lane == 0) {
#pragma unroll
            for (int component = 0; component < component_count; ++component)
                atomicAdd(&sums[component], local[component]);
        }
    }
}

// Solve the 3-by-3 normal equations on one thread using pivoted Gauss-Jordan elimination.
__global__ void solve_regression(const double* sums, double* coefficients, int* fitted) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;  // This tiny solve is intentionally serial.
    if (sums[9] < 3.0) { *fitted = 0; return; }       // Need at least three observations.
    // Form the augmented system X'X * beta = X'Y.
    double a[3][4] = {{sums[0], sums[1], sums[2], sums[6]},
                      {sums[1], sums[3], sums[4], sums[7]},
                      {sums[2], sums[4], sums[5], sums[8]}};
    for (int col = 0; col < 3; ++col) {
        // Pick the largest available pivot to improve numerical stability.
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
            if (fabs(a[row][col]) > fabs(a[pivot][col])) pivot = row;
        if (fabs(a[pivot][col]) < 1e-12) { *fitted = 0; return; }  // Singular regression.
        // Swap, normalise, then eliminate this pivot column from the other rows.
        for (int entry = col; entry < 4; ++entry) {
            const double tmp = a[col][entry]; a[col][entry] = a[pivot][entry]; a[pivot][entry] = tmp;
        }
        const double divisor = a[col][col];
        for (int entry = col; entry < 4; ++entry) a[col][entry] /= divisor;
        for (int row = 0; row < 3; ++row) if (row != col) {
            const double factor = a[row][col];
            for (int entry = col; entry < 4; ++entry) a[row][entry] -= factor * a[col][entry];
        }
    }
    coefficients[0] = a[0][3]; coefficients[1] = a[1][3]; coefficients[2] = a[2][3];
    *fitted = 1;  // Signals that the policy kernel can use these coefficients.
}

// Replace a later stopping date when immediate exercise beats estimated continuation.
template <typename PathValue>
__global__ void apply_exercise_policy(const PathValue* spot_row, double* cashflows,
                                      std::size_t* exercise_indices, const double* coefficients,
                                      const int* fitted, std::size_t paths_count,
                                      std::size_t time_index, double strike) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths_count || *fitted == 0) return;  // Keep existing exercise choice if fit failed.
    const double spot = static_cast<double>(spot_row[path]);
    const double payoff = fmax(strike - spot, 0.0);
    if (payoff <= 0.0) return;
    const double x = spot / strike;
    const double continuation = coefficients[0] + coefficients[1] * x + coefficients[2] * x * x;
    if (payoff > continuation) { exercise_indices[path] = time_index; cashflows[path] = payoff; }
}

// Reduce discounted pathwise payoffs into their first and second moments.
__global__ void accumulate_price_statistics(const double* cashflows,
                                            const std::size_t* exercise_indices, double* sums,
                                            std::size_t paths_count, double rate_dt) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths_count) return;
    const double value = cashflows[path] * exp(-rate_dt * static_cast<double>(exercise_indices[path]));
    atomicAdd(&sums[0], value); atomicAdd(&sums[1], value * value);
}

// =============================================================================
// Stage 3: expected-exposure and CVA calculation
// =============================================================================

// Add a live option's value at one grid date to the total exposure at that date.
__global__ void accumulate_exposure(const double* cashflows, const std::size_t* exercise_indices,
                                    double* exposure, std::size_t paths_count, std::size_t time_index,
                                    double rate_dt) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path < paths_count && time_index <= exercise_indices[path]) {
        // Value the chosen future cashflow back to this time point.
        atomicAdd(exposure, cashflows[path] * exp(-rate_dt *
            static_cast<double>(exercise_indices[path] - time_index)));
    }
}

// Integrate the exposure profile against default probabilities under a constant hazard rate.
__global__ void calculate_cva_kernel(const double* exposure, const std::size_t* exposure_indices,
                                     std::size_t exposure_count, double* cva,
                                     double dt, double rate, double hazard, double recovery) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;  // Short time-loop: one thread is sufficient.
    double value = 0.0;
    std::size_t previous_index = 0;
    for (std::size_t i = 0; i < exposure_count; ++i) {
        const std::size_t time_index = exposure_indices[i];
        if (time_index == 0) continue;
        const double t = dt * static_cast<double>(time_index);
        const double previous = dt * static_cast<double>(previous_index);
        // Change in survival probability is the probability of default in this interval.
        const double default_increment = exp(-hazard * previous) - exp(-hazard * t);
        value += exp(-rate * t) * exposure[i] * default_increment * (1.0 - recovery);
        previous_index = time_index;
    }
    *cva = value;
}

// Turn sums across Monte Carlo paths into expected exposures.
__global__ void normalize_exposure(double* exposure, std::size_t time_count, double path_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < time_count) exposure[index] /= path_count;
}

// Return elapsed GPU-event time in milliseconds, checking the CUDA call centrally.
float elapsed_ms(cudaEvent_t first, cudaEvent_t last) {
    float elapsed{};
    check_cuda(cudaEventElapsedTime(&elapsed, first, last), "cudaEventElapsedTime");
    return elapsed;
}

}  // namespace

// =============================================================================
// Public CUDA backend
// =============================================================================

// CUDA requires a visible device with compute capability 7+ for this pipeline.
bool cuda_backend_available() noexcept {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, 0) == cudaSuccess && properties.major >= 7;
}

// Give callers a readable device name, or explain why the backend cannot run.
std::string cuda_backend_description() {
    if (!cuda_backend_available()) return "no compatible CUDA device";
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    return properties.name;
}

// Run the complete GPU workflow: paths -> Bermudan LSM -> exposure -> CVA.
template <typename PathValue>
CudaBermudanResult run_cuda_bermudan_xva_impl(const MarketParams& market, const BermudanPutParams& option,
                                              const SimulationConfig& simulation, const CreditParams& credit,
                                              const CudaPipelineOptions& pipeline_options) {
    // Validate before consuming device memory or starting asynchronous GPU work.
    validate(market); validate(simulation); validate(option, simulation); validate(credit);
    if (!cuda_backend_available()) throw std::runtime_error("no compatible CUDA device is available");

    // Each one-dimensional kernel uses 256 threads; blocks round up to cover all paths.
    constexpr int threads = 256;
    const int blocks = static_cast<int>((simulation.num_paths + threads - 1) / threads);
    const std::size_t time_count = simulation.num_time_steps + 1;  // Includes the time-zero row.
    // Precompute the same exact-GBM drift and diffusion coefficients as the CPU implementation.
    const double dt = option.maturity / static_cast<double>(simulation.num_time_steps);
    const double rate_dt = market.rate * dt;
    const double drift = (market.rate - market.dividend_yield - 0.5 * market.volatility * market.volatility) * dt;
    const double diffusion = market.volatility * std::sqrt(dt);

    // Allocate all device-resident data once.  paths is laid out as [time][path].
    DeviceBuffer<PathValue> paths(time_count * simulation.num_paths);
    DeviceBuffer<double> cashflows(simulation.num_paths);
    DeviceBuffer<std::size_t> exercise_indices(simulation.num_paths);
    DeviceBuffer<double> regression_sums(10), coefficients(3), price_sums(2);
    DeviceBuffer<int> fitted(1);
    std::vector<std::size_t> host_exposure_indices;
    if (pipeline_options.cva_exercise_dates_only) {
        host_exposure_indices = option.exercise_indices;
    } else {
        host_exposure_indices.resize(time_count);
        for (std::size_t i = 0; i < time_count; ++i) host_exposure_indices[i] = i;
    }
    const std::size_t exposure_count = host_exposure_indices.size();
    DeviceBuffer<double> exposure(exposure_count), cva(1);
    DeviceBuffer<std::size_t> exposure_indices_device(exposure_count);
    check_cuda(cudaMemcpy(exposure_indices_device.data(), host_exposure_indices.data(),
        exposure_indices_device.bytes(), cudaMemcpyHostToDevice), "copy exposure indices");
    // Events delimit the three high-level GPU stages for the timing report.
    cudaEvent_t start{}, after_paths{}, after_lsm{}, end{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate"); check_cuda(cudaEventCreate(&after_paths), "cudaEventCreate");
    check_cuda(cudaEventCreate(&after_lsm), "cudaEventCreate"); check_cuda(cudaEventCreate(&end), "cudaEventCreate");
    check_cuda(cudaEventRecord(start), "cudaEventRecord");

    // -------------------------------------------------------------------------
    // 1. Generate all GBM paths and initialise cashflows at maturity.
    // -------------------------------------------------------------------------
    generate_gbm_paths<<<blocks, threads>>>(paths.data(), simulation.num_paths,
        simulation.num_time_steps, simulation.seed, market.spot, drift, diffusion);
    // The final exercise date is maturity, so every path begins with that stopping rule.
    initialise_terminal_payoffs<<<blocks, threads>>>(paths.data() + option.exercise_indices.back() * simulation.num_paths,
        cashflows.data(), exercise_indices.data(), simulation.num_paths, option.exercise_indices.back(), option.strike);
    check_cuda(cudaGetLastError(), "path-generation kernel launch");
    check_cuda(cudaEventRecord(after_paths), "cudaEventRecord");

    // -------------------------------------------------------------------------
    // 2. Work backwards over earlier exercise dates using Longstaff-Schwartz.
    // -------------------------------------------------------------------------
    for (std::size_t pos = option.exercise_indices.size() - 1; pos-- > 0;) {
        const std::size_t time_index = option.exercise_indices[pos];
        // Every exercise date gets its own zeroed normal-equation accumulator.
        check_cuda(cudaMemset(regression_sums.data(), 0, regression_sums.bytes()), "cudaMemset regression sums");
        accumulate_regression<<<blocks, threads>>>(paths.data() + time_index * simulation.num_paths,
            cashflows.data(), exercise_indices.data(), regression_sums.data(), simulation.num_paths,
            time_index, option.strike, rate_dt);
        // Fit continuation, then use it to replace later exercise when appropriate.
        solve_regression<<<1, 1>>>(regression_sums.data(), coefficients.data(), fitted.data());
        apply_exercise_policy<<<blocks, threads>>>(paths.data() + time_index * simulation.num_paths,
            cashflows.data(), exercise_indices.data(), coefficients.data(), fitted.data(), simulation.num_paths,
            time_index, option.strike);
    }
    // Reduce final stopping decisions to price moments for mean and standard error.
    check_cuda(cudaMemset(price_sums.data(), 0, price_sums.bytes()), "cudaMemset price sums");
    accumulate_price_statistics<<<blocks, threads>>>(cashflows.data(), exercise_indices.data(), price_sums.data(),
        simulation.num_paths, rate_dt);
    check_cuda(cudaGetLastError(), "LSM kernel launch");
    check_cuda(cudaEventRecord(after_lsm), "cudaEventRecord");

    // -------------------------------------------------------------------------
    // 3. Build expected exposure at each date, then calculate CVA.
    // -------------------------------------------------------------------------
    check_cuda(cudaMemset(exposure.data(), 0, exposure.bytes()), "cudaMemset exposure");
    for (std::size_t position = 0; position < exposure_count; ++position) {
        // One kernel launch accumulates all paths into the selected exposure date.
        accumulate_exposure<<<blocks, threads>>>(cashflows.data(), exercise_indices.data(), exposure.data() + position,
            simulation.num_paths, host_exposure_indices[position], rate_dt);
    }
    normalize_exposure<<<static_cast<int>((exposure_count + threads - 1) / threads), threads>>>(
        exposure.data(), exposure_count, static_cast<double>(simulation.num_paths));
    calculate_cva_kernel<<<1, 1>>>(exposure.data(), exposure_indices_device.data(), exposure_count, cva.data(), dt,
        market.rate, credit.hazard_rate, credit.recovery_rate);
    check_cuda(cudaGetLastError(), "CVA kernel launch");
    check_cuda(cudaEventRecord(end), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize");  // Results are now safe to read on host.

    // -------------------------------------------------------------------------
    // 4. Copy outputs to the CPU and construct the public result objects.
    // -------------------------------------------------------------------------
    double host_price_sums[2]{}; double host_cva{};
    std::vector<double> host_exposure(exposure_count);
    std::vector<std::size_t> host_exercise_indices(simulation.num_paths);
    std::vector<double> host_cashflows(simulation.num_paths);
    check_cuda(cudaMemcpy(host_price_sums, price_sums.data(), sizeof(host_price_sums), cudaMemcpyDeviceToHost), "copy price sums");
    check_cuda(cudaMemcpy(&host_cva, cva.data(), sizeof(host_cva), cudaMemcpyDeviceToHost), "copy CVA");
    check_cuda(cudaMemcpy(host_exposure.data(), exposure.data(), exposure.bytes(), cudaMemcpyDeviceToHost), "copy exposure");
    check_cuda(cudaMemcpy(host_exercise_indices.data(), exercise_indices.data(), exercise_indices.bytes(), cudaMemcpyDeviceToHost), "copy exercise indices");
    check_cuda(cudaMemcpy(host_cashflows.data(), cashflows.data(), cashflows.bytes(), cudaMemcpyDeviceToHost), "copy exercise cashflows");
    // Derive mean, unbiased sample variance, standard error, and the 95% interval.
    const double count = static_cast<double>(simulation.num_paths);
    const double price = host_price_sums[0] / count;
    const double variance = simulation.num_paths > 1
        ? std::max((host_price_sums[1] - host_price_sums[0] * price) / (count - 1.0), 0.0) : 0.0;
    const double standard_error = std::sqrt(variance / count);
    constexpr double z95 = 1.959963984540054;
    // Attribute the recorded GPU elapsed time to the path, LSM, and CVA stages.
    const StageTimings path_timings{0.0, static_cast<double>(elapsed_ms(start, after_paths)), 0.0, 0.0,
        static_cast<double>(elapsed_ms(start, after_paths))};
    const StageTimings lsm_timings{0.0, 0.0, 0.0, 0.0,
        static_cast<double>(elapsed_ms(after_paths, after_lsm))};
    const StageTimings cva_timings{0.0, 0.0, 0.0, 0.0,
        static_cast<double>(elapsed_ms(after_lsm, end))};
    // CUDA events are raw handles, so clean them up once their timings have been read.
    cudaEventDestroy(start); cudaEventDestroy(after_paths); cudaEventDestroy(after_lsm); cudaEventDestroy(end);

    // Attach each expected-exposure point to its corresponding simulation time.
    ExposureProfile profile;
    profile.times.resize(exposure_count); profile.expected_exposure = std::move(host_exposure);
    for (std::size_t i = 0; i < exposure_count; ++i)
        profile.times[i] = dt * static_cast<double>(host_exposure_indices[i]);
    // Return pricing diagnostics, CVA/exposure outputs, timings, allocation size, and device name.
    const std::size_t workspace_bytes = paths.bytes() + cashflows.bytes() + exercise_indices.bytes()
        + regression_sums.bytes() + coefficients.bytes() + price_sums.bytes() + fitted.bytes()
        + exposure.bytes() + exposure_indices_device.bytes() + cva.bytes();
    return {{price, standard_error, price - z95 * standard_error, price + z95 * standard_error,
             std::move(host_exercise_indices), std::move(host_cashflows), lsm_timings},
            {host_cva, std::move(profile), cva_timings}, path_timings, paths.bytes(), workspace_bytes,
            cuda_backend_description()};
}

CudaBermudanResult run_cuda_bermudan_xva(const MarketParams& market, const BermudanPutParams& option,
                                         const SimulationConfig& simulation, const CreditParams& credit,
                                         const CudaPipelineOptions& pipeline_options) {
    if (pipeline_options.float_paths)
        return run_cuda_bermudan_xva_impl<float>(market, option, simulation, credit, pipeline_options);
    return run_cuda_bermudan_xva_impl<double>(market, option, simulation, credit, pipeline_options);
}

CudaBermudanResult run_cuda_bermudan_xva(const MarketParams& market, const BermudanPutParams& option,
                                         const SimulationConfig& simulation, const CreditParams& credit) {
    return run_cuda_bermudan_xva(market, option, simulation, credit, {});
}

}  // namespace gpu_lsm
