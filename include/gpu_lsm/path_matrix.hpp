#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gpu_lsm {

// Dense time-major storage: states[time_index * num_paths + path_index].
class PathMatrix {
public:
    PathMatrix(std::size_t num_times, std::size_t num_paths)
        : num_times_(num_times),
          num_paths_(num_paths),
          states_(checked_size(num_times, num_paths)) {
        if (num_times == 0 || num_paths == 0) {
            throw std::invalid_argument("PathMatrix dimensions must be positive");
        }
    }

    [[nodiscard]] double& operator()(std::size_t time, std::size_t path) {
        return states_.at(index(time, path));
    }

    [[nodiscard]] const double& operator()(std::size_t time, std::size_t path) const {
        return states_.at(index(time, path));
    }

    [[nodiscard]] std::size_t num_times() const noexcept { return num_times_; }
    [[nodiscard]] std::size_t num_paths() const noexcept { return num_paths_; }
    [[nodiscard]] std::size_t size() const noexcept { return states_.size(); }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return states_.size() * sizeof(double);
    }

    [[nodiscard]] double* data() noexcept { return states_.data(); }
    [[nodiscard]] const double* data() const noexcept { return states_.data(); }

private:
    static std::size_t checked_size(std::size_t num_times, std::size_t num_paths) {
        if (num_times != 0 && num_paths > static_cast<std::size_t>(-1) / num_times) {
            throw std::length_error("PathMatrix dimensions overflow");
        }
        return num_times * num_paths;
    }

    [[nodiscard]] std::size_t index(std::size_t time, std::size_t path) const {
        if (time >= num_times_ || path >= num_paths_) {
            throw std::out_of_range("PathMatrix index out of range");
        }
        return time * num_paths_ + path;
    }

    std::size_t num_times_{};
    std::size_t num_paths_{};
    std::vector<double> states_;
};

}  // namespace gpu_lsm

