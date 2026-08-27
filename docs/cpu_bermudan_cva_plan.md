# CPU Bermudan LSM and Unilateral CVA Baseline

## Model

- Risk-neutral exact GBM with double-precision, time-major path storage.
- One-year Bermudan put with weekly simulation dates and quarterly exercise
  indices `13, 26, 39, 52`.
- Longstaff--Schwartz backwards induction on in-the-money paths using the
  quadratic normalised-spot basis `1, S/K, (S/K)^2`.
- Independent, flat counterparty hazard rate and recovery rate for unilateral
  CVA.

## CVA convention

For each simulated date, exposure is the non-negative remaining policy
cashflow discounted from its selected exercise date. Expected exposure is the
path mean. CVA is evaluated from discrete survival-probability increments:

```text
CVA = sum_i D(0,t_i) EE(t_i) [Q(t_{i-1}) - Q(t_i)] (1 - recovery)
Q(t) = exp(-hazard_rate * t)
```

The project intentionally excludes collateral, netting, portfolios, DVA, FVA,
KVA, and wrong-way risk.

## Validation and CUDA handoff

The CPU validator checks that the confidence-aware Bermudan LSM estimate is no
lower than the Black--Scholes European put and no higher than a high-step CRR
American put. Catch2 covers path generation, inputs, LSM policy, exposure/CVA,
validation, and a small benchmark smoke test.

CUDA must first reproduce the CPU double/full-path calculation. Later
experiments compare full-path storage with quarterly exercise-date storage,
reduced precision, and checkpoint/replay while reporting price/CVA error,
runtime, and device-memory reduction.
