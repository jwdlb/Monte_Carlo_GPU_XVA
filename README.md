# GPU LSM XVA

A C++20/CUDA research project for studying how GPU memory layout, batching and
numerical precision affect regression-based Bermudan-option exposure calculations.

The repository currently contains the **Phase 1 skeleton** only. It defines the
configuration, result, timing and time-major path-storage interfaces that the CPU
reference implementation will use. Numerical routines deliberately throw
`std::logic_error` until their Phase 1 implementations are added.

## Configure and build

```bash
cmake -S . -B build
cmake --build build
./build/gpu_lsm_xva
```

Tests use Catch2 3 and are optional at skeleton stage:

```bash
cmake -S . -B build -DGPU_LSM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Phase 1 implementation order

1. Implement and validate the analytical Black--Scholes put.
2. Implement terminal-only European Monte Carlo and its statistics.
3. Implement full-path exact GBM generation using `PathMatrix`.
4. Add Euler--Maruyama through the same path-generation interface.
5. Add moment validation, reproducibility checks and stage-level reporting.

CUDA sources are intentionally not part of the build until the CPU reference is
validated.

