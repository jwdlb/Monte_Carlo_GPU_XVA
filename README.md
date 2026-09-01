# GPU Bermudan LSM XVA

A C++20/CUDA implementation for pricing a Bermudan put under exact geometric
Brownian motion. It uses quadratic Longstaff--Schwartz Monte Carlo (LSM) for
exercise decisions and calculates unilateral CVA from the resulting exposure
profile. A CPU implementation is retained as a reference and fallback.

The default model uses 100,000 paths, 52 weekly time steps, quarterly Bermudan
exercise dates, a flat 2% counterparty hazard rate, and 40% recovery. It is
deliberately a small, validated baseline for CUDA and path-compression
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
path grid remains on the device in the selected `double` or `float` format;
only final pricing/CVA results and the exercise policy are transferred to the
host.

`validate` is the default command. It reports the LSM Bermudan price, its
standard error, unilateral CVA, full-path memory, and checks that the estimate
lies between the Black--Scholes European put and a CRR American-put benchmark.

`benchmark` runs the same calculation with 100,000, 500,000, and 1,000,000
paths, reporting memory and path/LSM/CVA timings.

## Path-compression experiment

The CUDA pipeline supports two optional reductions:

- **Exercise-date CVA:** calculate exposure only at the four dates on which the
  option can be exercised, instead of all 53 weekly grid points.
- **Float paths:** store the simulated path matrix in 32-bit `float` values
  instead of 64-bit `double` values. Regression, cashflows, price, and CVA
  accumulation remain in double precision.

Run all four configurations with:

```bash
./build-cuda/gpu_lsm_xva compression-benchmark 100000
```

### RTX 4060 results

Measured with 100,000 paths on an NVIDIA GeForce RTX 4060. Errors are relative
to the full-grid, double-path baseline.

| Configuration | Price error | CVA error | Path memory | Exposure dates | Total time | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Double paths, full CVA grid | 0.000000% | 0.000000% | 40.44 MiB | 53 | 10.53 ms | 1.00x |
| Double paths, exercise-date CVA | 0.000000% | 0.000000% | 40.44 MiB | 4 | 4.87 ms | 2.16x |
| Float paths, full CVA grid | 0.000000% | 0.000000% | 20.22 MiB | 53 | 10.61 ms | 0.99x |
| Float paths, exercise-date CVA | 0.000000% | 0.000000% | 20.22 MiB | 4 | 4.04 ms | 2.60x |

The combined configuration gave the best result: **50% less path memory** and
about **2.6x lower total runtime**, with no price or CVA difference visible at
six decimal places. Float storage alone reduced memory but did not improve
runtime on this GPU; the main speedup came from reducing the CVA exposure grid
from 53 dates to 4.

For this model, exercise decisions and cashflow changes occur only at Bermudan
exercise dates. This explains why grouping the weekly CVA intervals by those
dates produced the same reported CVA in this experiment. Timings are from one
machine and should be treated as hardware-specific rather than universal.

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
unilateral CVA, CPU and CUDA pipelines, exercise-date CVA, and selectable
double- or float-precision path storage.

Excluded: portfolios/netting, collateral, wrong-way risk, DVA, FVA, KVA, and
stochastic credit.
