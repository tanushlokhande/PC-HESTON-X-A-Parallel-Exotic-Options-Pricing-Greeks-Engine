# HESTON-X: A Parallel Exotic-Options Pricing & Greeks Engine

**Author:** Tanush Rajaram Lokhande | **Roll No:** 241080035
**Course:** Parallel Processing Lab

A parallel Monte Carlo engine that prices a book of path-dependent exotic options under the **Heston stochastic-volatility model** and computes risk sensitivities (Delta and sensitivity to initial variance). It compares serial, multiprocessing and Numba shared-memory implementations and analyses their scaling.

---

## Problem

Exotic options on an asset with stochastic volatility have no closed-form price, so they are priced by Monte Carlo. A single risk run needs 5 market scenarios x hundreds of thousands of paths x 252 time steps, which is slow. This project parallelises that workload and studies speedup, efficiency and scaling limits.

## Instruments priced

| Instrument | Payoff |
|---|---|
| European call | max(S_T - K, 0) |
| Arithmetic Asian call | max(average(S) - K, 0) |
| Up-and-out barrier call | max(S_T - K, 0) if S stays below barrier B, else 0 |
| Fixed lookback call | max(max(S) - K, 0) |

## Model

Heston dynamics, discretised with the full-truncation Euler scheme:

```
dS = (r - q) S dt + sqrt(v) S dW1
dv = kappa (theta - v) dt + xi sqrt(v) dW2        corr(dW1, dW2) = rho
```

Default parameters: S0=100, r=5%, q=1%, v0=0.04, kappa=2, theta=0.04, xi=0.5, rho=-0.7, T=1 year, 252 steps.

## Features

- Semi-analytic Heston European price (Fourier inversion, "little trap" formulation) used to validate the simulator.
- Antithetic variates for variance reduction.
- Independent parallel random streams via `SeedSequence.spawn`, so results do not depend on the number of workers.
- Four implementations:
  1. Serial NumPy baseline
  2. `multiprocessing` (data parallelism over path chunks)
  3. Two-level task parallelism (scenario x chunk) for Greeks
  4. Numba `prange` (fused, shared-memory kernel)
- Greeks by bump-and-reprice with common random numbers.
- Performance study: speedup, efficiency, Karp-Flatt serial fraction, Amdahl bound, plus a 1/sqrt(N) convergence plot.

## How to run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com).
2. Choose **File > Upload notebook** and select `HestonX_Parallel_Options_Engine.ipynb`.
3. Choose **Runtime > Run all**.

No installation is needed: NumPy, SciPy, Matplotlib and Numba are preinstalled on Colab. The default sizes finish in a few minutes.

To change the workload, edit these in the notebook:

| Variable | Meaning | Default |
|---|---|---|
| `N_PAIRS` | antithetic pairs for pricing | 200,000 |
| `N_CHUNKS` | number of work chunks | 32 |
| `BENCH_PAIRS` | pairs used in the scaling study | 100,000 |
| `N_STEPS` | monitoring dates | 252 |

## Notebook structure

| Section | Content |
|---|---|
| 1 | Semi-analytic Heston European call (ground truth) |
| 2 | NumPy and Numba Monte Carlo kernels |
| 3 | Parallel drivers with reproducible RNG |
| 4 | Correctness check against the analytic price |
| 5 | Two-level parallel Greeks |
| 6 | Strong-scaling study: speedup, efficiency, Karp-Flatt |
| 7 | Convergence plot |

## Expected outputs

- A price table for the four instruments from both the multiprocessing and Numba engines, with standard errors.
- Analytic European price of about 9.44, with the Monte Carlo estimate within about 3 standard errors.
- A Price / Delta / dP/dV0 table for the book.
- Speedup and efficiency plots, and a convergence plot.

Sanity checks built in: the knock-out price must not exceed the vanilla price, and the lookback price must not be below it.

## Notes and limitations

- Free Colab usually provides about 2 CPU cores, so speedups are modest (around 2x at best). The scaling study also runs oversubscribed worker counts to show where scaling stops.
- Numba's speedup over the NumPy baseline includes single-thread gains (loop fusion, no temporaries), not only parallelism.
- The barrier option's bumped Delta is noisy because its payoff is discontinuous.
- Discretisation: full-truncation Euler has small bias; barrier monitoring is discrete (daily).

## Possible extensions

- American options via Longstaff-Schwartz (parallel regression)
- Portfolio VaR / CVaR with nested simulation
- GPU port with `numba.cuda` or CuPy
- Pathwise or likelihood-ratio Greeks
- Weak-scaling study

## References

1. S. L. Heston, "A closed-form solution for options with stochastic volatility," *Review of Financial Studies*, 1993.
2. P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.
3. R. Lord, R. Koekkoek, D. van Dijk, "A comparison of biased simulation schemes for stochastic volatility models," *Quantitative Finance*, 2010.
4. H. Albrecher et al., "The little Heston trap," *Wilmott Magazine*, 2007.
5. G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities," 1967.
6. A. H. Karp, H. P. Flatt, "Measuring parallel processor performance," *CACM*, 1990.
7. S. K. Lam, A. Pitrou, S. Seibert, "Numba: a LLVM-based Python JIT compiler," 2015.
