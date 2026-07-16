#include "gpu_lsm/statistics.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("95 percent interval is symmetric") {
    const auto interval = gpu_lsm::confidence_interval_95(10.0, 2.0);
    REQUIRE((10.0 - interval.low) == Catch::Approx(interval.high - 10.0));
}

TEST_CASE("sample statistics match a small known sample") {
    SKIP("Enable after implementing summarize");
}

