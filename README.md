# CPU Bermudan LSM XVA

A C++20 CPU reference implementation for a single Bermudan put under exact
geometric Brownian motion. It uses quadratic-basis Longstaff--Schwartz Monte
Carlo (LSM) for exercise decisions and calculates unilateral CVA from the
resulting positive-exposure profile.

The default model uses 100,000 paths, 52 weekly time steps, quarterly Bermudan
exercise dates, a flat 2% counterparty hazard rate, and 40% recovery. It is
deliberately a small, validated baseline for later CUDA and path-compression
experiments.

## Build and run

```bash
cmake -S . -B build
cmake --build build --parallel
./build/gpu_lsm_xva validate
./build/gpu_lsm_xva benchmark
```

## CUDA backend

The default build is CPU-only and works without an NVIDIA toolkit. To build
the optional CUDA backend, configure with a CUDA toolkit and a host compiler
supported by that toolkit:

```bash
cmake -S . -B build-cuda -DGPU_LSM_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
./build-cuda/gpu_lsm_xva validate
```

At runtime the executable automatically uses a Volta-or-newer CUDA device when
one is available; otherwise it runs the existing CPU implementation. The CUDA
path retains the full double-precision path grid on the device and transfers
only final pricing/CVA results and the exercise policy to the host.

`validate` is the default command. It reports the LSM Bermudan price, its
standard error, unilateral CVA, full-path memory, and checks that the estimate
lies between the Black--Scholes European put and a CRR American-put benchmark.

`benchmark` runs the same calculation with 100,000, 500,000, and 1,000,000
paths, reporting memory and path/LSM/CVA timings.

## Tests

Tests use Catch2 v3.15. CMake first looks for a system package; otherwise it
fetches the pinned `v3.15.0` release. Run them with:

```bash
ctest --test-dir build --output-on-failure
```

For an offline application-only build, disable tests:

```bash
cmake -S . -B build-offline -DGPU_LSM_BUILD_TESTS=OFF
cmake --build build-offline --parallel
```

## Scope

Included: a single uncollateralised Bermudan put, independent flat credit,
unilateral CVA, and full double-precision path storage.

Excluded: CUDA, compression, portfolios/netting, collateral, wrong-way risk,
DVA, FVA, KVA, and stochastic credit. CUDA work will first reproduce this
baseline, then compare full-path, exercise-date-only, and reduced-precision
storage strategies.
