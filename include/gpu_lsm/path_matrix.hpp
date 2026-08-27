#pragma once

/*
 * Purpose: Owns a dense two-dimensional matrix of simulated share prices.
 * Storage is time-major: all paths at one simulation time are contiguous.
 * That layout makes date-by-date exposure work efficient and GPU-friendly.
 */

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace gpu_lsm {

// Dense time-major storage: states[time_index * num_paths + path_index].
class PathMatrix {
public:
    // Allocate one price for every (time, path) pair after validating dimensions.
    // This deliberately uses a simple constructor body rather than a member
    // initialiser list so the three setup steps are easy to follow.
    PathMatrix(std::size_t num_times, std::size_t num_paths) {
        num_times_ = num_times;
        num_paths_ = num_paths;
        states_ = std::vector<double>(checked_size(num_times, num_paths));
    }

    // Mutable, bounds-checked access to one simulated state.
    [[nodiscard]] double& operator()(std::size_t time, std::size_t path) {
        return states_.at(index(time, path));
    }

    // Read-only counterpart, usable when the matrix itself is const.
    [[nodiscard]] const double& operator()(std::size_t time, std::size_t path) const {
        return states_.at(index(time, path));
    }

    // Basic shape and raw-storage queries; they cannot throw.
    [[nodiscard]] std::size_t num_times() const noexcept { return num_times_; }
    [[nodiscard]] std::size_t num_paths() const noexcept { return num_paths_; }
    [[nodiscard]] std::size_t size() const noexcept { return states_.size(); }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return states_.size() * sizeof(double);
    }

    // Direct contiguous storage access for high-performance interoperation.
    [[nodiscard]] double* data() noexcept { return states_.data(); }
    [[nodiscard]] const double* data() const noexcept { return states_.data(); }

    // Non-owning mutable view of all paths at one simulation time.
    [[nodiscard]] std::span<double> time_slice(std::size_t time) {
        if (time >= num_times_) {
            throw std::out_of_range("PathMatrix time index out of range");
        }
        return {states_.data() + time * num_paths_, num_paths_};
    }

    // Read-only version of time_slice; no path values may be changed through it.
    [[nodiscard]] std::span<const double> time_slice(std::size_t time) const {
        if (time >= num_times_) {
            throw std::out_of_range("PathMatrix time index out of range");
        }
        return {states_.data() + time * num_paths_, num_paths_};
    }

private:
    // Reject empty matrices and multiplication overflow before vector allocation.
    static std::size_t checked_size(std::size_t num_times, std::size_t num_paths) {
        if (num_times == 0 || num_paths == 0) {
            throw std::invalid_argument("PathMatrix dimensions must be positive");
        }
        if (num_paths > std::numeric_limits<std::size_t>::max() / num_times) {
            throw std::length_error("PathMatrix dimensions overflow");
        }
        return num_times * num_paths;
    }

    // Convert a checked (time, path) coordinate to the flat vector index.
    [[nodiscard]] std::size_t index(std::size_t time, std::size_t path) const {
        if (time >= num_times_ || path >= num_paths_) {
            throw std::out_of_range("PathMatrix index out of range");
        }
        return time * num_paths_ + path;
    }

    // Matrix shape followed by the contiguous owning storage.
    std::size_t num_times_{};
    std::size_t num_paths_{};
    std::vector<double> states_;
};

}  // namespace gpu_lsm
