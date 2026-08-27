// GPU implementation of the full Bermudan-put pricing and CVA workflow.
//
// The three main stages are: GBM path generation, Longstaff-Schwartz exercise
// regression, and exposure/CVA calculation.  Keeping them together makes the
// device-resident data flow explicit and avoids unnecessary host transfers.
#include "gpu_lsm/cuda_backend.hpp"

#include <cuda_runtime.h>
#include <curand.h>

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

// cuRAND has no equivalent error-string API, so retain the failed operation name.
void check_curand(curandStatus_t status, const char* operation) {
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
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

// RAII wrapper around the seeded cuRAND generator used for GBM normal shocks.
class CurandGenerator {
public:
    explicit CurandGenerator(std::uint64_t seed) {
        // Philox is a pseudo-random generator designed for massively parallel work.
        check_curand(curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10),
                     "curandCreateGenerator");
        check_curand(curandSetPseudoRandomGeneratorSeed(generator_, seed),
                     "curandSetPseudoRandomGeneratorSeed");
    }
    ~CurandGenerator() { if (generator_ != nullptr) curandDestroyGenerator(generator_); }
    // Lets cuRAND fill a supplied device buffer while this class retains ownership.
    curandGenerator_t get() const noexcept { return generator_; }
private:
    curandGenerator_t generator_{};
};

// =============================================================================
// Stage 1: exact geometric-Brownian-motion path generation
// =============================================================================

// Set the first path-matrix row to S(0), using one GPU thread per path.
__global__ void set_initial_spot(double* paths, std::size_t paths_count, double spot) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path < paths_count) paths[path] = spot;  // Guard threads in the rounded-up final block.
}

// Advance every path by one exact GBM interval:
// S(t + dt) = S(t) * exp(drift + diffusion * standard_normal_shock).
__global__ void evolve_gbm(const double* previous, const double* normals, double* current,
                           std::size_t paths_count, double drift, double diffusion) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path < paths_count) current[path] = previous[path] * exp(drift + diffusion * normals[path]);
}

// At maturity a Bermudan put is settled for max(K - S(T), 0) on each path.
__global__ void initialise_terminal_payoffs(const double* terminal, double* cashflows,
                                             std::size_t* exercise_indices, std::size_t paths_count,
                                             std::size_t maturity_index, double strike) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path < paths_count) {
        cashflows[path] = fmax(strike - terminal[path], 0.0);  // Intrinsic put payoff.
        exercise_indices[path] = maturity_index;               // Initial stopping date is maturity.
    }
}

// =============================================================================
// Stage 2: Longstaff-Schwartz regression and early-exercise policy
// =============================================================================

// sums stores six symmetric X'X terms, three X'Y terms, then the ITM count.
__global__ void accumulate_regression(const double* spot_row, const double* cashflows,
                                      const std::size_t* exercise_indices, double* sums,
                                      std::size_t paths_count, std::size_t time_index,
                                      double strike, double rate_dt) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths_count) return;  // Ignore padding threads.
    const double payoff = fmax(strike - spot_row[path], 0.0);
    if (payoff <= 0.0) return;  // LSM fits continuation only to in-the-money paths.
    // Use the quadratic basis [1, x, x^2], with spot normalised by strike.
    const double x = spot_row[path] / strike;
    const double x2 = x * x;
    // Discount the path's later chosen cashflow back to this potential exercise date.
    const double response = cashflows[path] * exp(-rate_dt *
        static_cast<double>(exercise_indices[path] - time_index));
    // Atomic additions combine each path's normal-equation contribution safely.
    atomicAdd(&sums[0], 1.0); atomicAdd(&sums[1], x);  atomicAdd(&sums[2], x2);
    atomicAdd(&sums[3], x2);  atomicAdd(&sums[4], x * x2); atomicAdd(&sums[5], x2 * x2);
    atomicAdd(&sums[6], response); atomicAdd(&sums[7], x * response);
    atomicAdd(&sums[8], x2 * response); atomicAdd(&sums[9], 1.0);
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
__global__ void apply_exercise_policy(const double* spot_row, double* cashflows,
                                      std::size_t* exercise_indices, const double* coefficients,
                                      const int* fitted, std::size_t paths_count,
                                      std::size_t time_index, double strike) {
    const auto path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= paths_count || *fitted == 0) return;  // Keep existing exercise choice if fit failed.
    const double payoff = fmax(strike - spot_row[path], 0.0);
    if (payoff <= 0.0) return;
    const double x = spot_row[path] / strike;
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
__global__ void calculate_cva_kernel(const double* exposure, double* cva, std::size_t time_steps,
                                     double dt, double rate, double hazard, double recovery) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;  // Short time-loop: one thread is sufficient.
    double value = 0.0;
    for (std::size_t i = 1; i <= time_steps; ++i) {
        const double t = dt * static_cast<double>(i);
        const double previous = dt * static_cast<double>(i - 1);
        // Change in survival probability is the probability of default in this interval.
        const double default_increment = exp(-hazard * previous) - exp(-hazard * t);
        value += exp(-rate * t) * exposure[i] * default_increment * (1.0 - recovery);
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
CudaBermudanResult run_cuda_bermudan_xva(const MarketParams& market, const BermudanPutParams& option,
                                         const SimulationConfig& simulation, const CreditParams& credit) {
    // Validate before consuming device memory or starting asynchronous GPU work.
    validate(market); validate(simulation); validate(option, simulation); validate(credit);
    if (!cuda_backend_available()) throw std::runtime_error("no compatible CUDA device is available");

    // Each one-dimensional kernel uses 256 threads; blocks round up to cover all paths.
    constexpr int threads = 256;
    const int blocks = static_cast<int>((simulation.num_paths + threads - 1) / threads);
    const std::size_t time_count = simulation.num_time_steps + 1;  // Includes the time-zero row.
    // cuRAND double-normal generation needs an even length, so add at most one unused value.
    const std::size_t normal_count = (simulation.num_paths + 1U) & ~std::size_t{1U};

    // Precompute the same exact-GBM drift and diffusion coefficients as the CPU implementation.
    const double dt = option.maturity / static_cast<double>(simulation.num_time_steps);
    const double rate_dt = market.rate * dt;
    const double drift = (market.rate - market.dividend_yield - 0.5 * market.volatility * market.volatility) * dt;
    const double diffusion = market.volatility * std::sqrt(dt);

    // Allocate all device-resident data once.  paths is laid out as [time][path].
    DeviceBuffer<double> paths(time_count * simulation.num_paths);
    DeviceBuffer<double> normals(normal_count);
    DeviceBuffer<double> cashflows(simulation.num_paths);
    DeviceBuffer<std::size_t> exercise_indices(simulation.num_paths);
    DeviceBuffer<double> regression_sums(10), coefficients(3), price_sums(2);
    DeviceBuffer<int> fitted(1);
    DeviceBuffer<double> exposure(time_count), cva(1);
    CurandGenerator generator(simulation.seed);
    // Events delimit the three high-level GPU stages for the timing report.
    cudaEvent_t start{}, after_paths{}, after_lsm{}, end{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate"); check_cuda(cudaEventCreate(&after_paths), "cudaEventCreate");
    check_cuda(cudaEventCreate(&after_lsm), "cudaEventCreate"); check_cuda(cudaEventCreate(&end), "cudaEventCreate");
    check_cuda(cudaEventRecord(start), "cudaEventRecord");

    // -------------------------------------------------------------------------
    // 1. Generate all GBM paths and initialise cashflows at maturity.
    // -------------------------------------------------------------------------
    set_initial_spot<<<blocks, threads>>>(paths.data(), simulation.num_paths, market.spot);
    for (std::size_t time = 1; time <= simulation.num_time_steps; ++time) {
        // Generate one standard-normal shock per path for this interval.
        check_curand(curandGenerateNormalDouble(generator.get(), normals.data(), normal_count, 0.0, 1.0),
                     "curandGenerateNormalDouble");
        // Evolve the previous row directly into the next row of the device path matrix.
        evolve_gbm<<<blocks, threads>>>(paths.data() + (time - 1) * simulation.num_paths, normals.data(),
                                        paths.data() + time * simulation.num_paths, simulation.num_paths,
                                        drift, diffusion);
    }
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
    for (std::size_t time = 0; time <= simulation.num_time_steps; ++time)
        // One kernel launch accumulates all paths into the selected exposure date.
        accumulate_exposure<<<blocks, threads>>>(cashflows.data(), exercise_indices.data(), exposure.data() + time,
            simulation.num_paths, time, rate_dt);
    normalize_exposure<<<static_cast<int>((time_count + threads - 1) / threads), threads>>>(
        exposure.data(), time_count, static_cast<double>(simulation.num_paths));
    calculate_cva_kernel<<<1, 1>>>(exposure.data(), cva.data(), simulation.num_time_steps, dt,
        market.rate, credit.hazard_rate, credit.recovery_rate);
    check_cuda(cudaGetLastError(), "CVA kernel launch");
    check_cuda(cudaEventRecord(end), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize");  // Results are now safe to read on host.

    // -------------------------------------------------------------------------
    // 4. Copy outputs to the CPU and construct the public result objects.
    // -------------------------------------------------------------------------
    double host_price_sums[2]{}; double host_cva{};
    std::vector<double> host_exposure(time_count);
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
    profile.times.resize(time_count); profile.expected_exposure = std::move(host_exposure);
    for (std::size_t i = 0; i < time_count; ++i) profile.times[i] = dt * static_cast<double>(i);
    // Return pricing diagnostics, CVA/exposure outputs, timings, allocation size, and device name.
    return {{price, standard_error, price - z95 * standard_error, price + z95 * standard_error,
             std::move(host_exercise_indices), std::move(host_cashflows), lsm_timings},
            {host_cva, std::move(profile), cva_timings}, path_timings, paths.bytes(), cuda_backend_description()};
}

}  // namespace gpu_lsm
