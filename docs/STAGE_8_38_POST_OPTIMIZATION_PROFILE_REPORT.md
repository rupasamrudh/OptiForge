# Stage 8.38 Report: Post-Optimization Bottleneck Profiling

**Date:** September 29, 2026  
**Status:** COMPLETE & VERIFIED  
**Repository State:** 1040 tests passing (0 failures, 19 benign warnings)

---

## 1. Executive Summary

Stage 8.37 introduced the controlled triangular solve provider architecture with pure Python/NumPy default substitution (`provider="python"`) and an optional Level-2 BLAS SciPy provider (`provider="scipy"`), delivering up to ~3.73x end-to-end acceleration on Dual Revised Simplex PFI workloads.

In accordance with strict numerical software engineering standards, **historical profiling data prior to Stage 8.37 can no longer be assumed to represent the solver's true runtime bottleneck distribution.**

Stage 8.38 executed an observational performance audit and profiling of the current OptiForge solver. A lightweight, zero-overhead profiling instrumentation system was designed, implemented, and validated across standard problem dimensions ($30\times 60$, $60\times 120$, $100\times 200$, $200\times 400$, and representative MILP branch-and-bound workloads).

### Key Empirical Findings:
1. **Under Default Python Provider (`triangular_solve_provider="python"`):**
   - Forward and backward triangular substitution across solver phases (`primal_solve`, `btran_v`, `pfi_update`) continues to consume **75% to 85%** of total solver runtime.
2. **Under Optional SciPy Provider (`triangular_solve_provider="scipy"`):**
   - End-to-end speedups range from **1.63x** ($30\times 60$) to **3.73x** ($200\times 400$).
   - The triangular solve bottleneck has been compressed, uncovering a new bottleneck distribution:
     - **Top Bottleneck 1 (`pfi_update`):** 33.4% of total runtime on $200\times 400$ ($S_{max} = 1.50x$).
     - **Top Bottleneck 2 (`initial_factorization`):** 21.1% to 33.3% of total runtime across $100\times 200$ and $200\times 400$ ($S_{max} = 1.27x$ to $1.50x$).
     - **Top Bottleneck 3 (`primal_solve`):** 19.8% to 21.3% of total runtime ($S_{max} = 1.25x$ to $1.27x$).
3. **Statistical `cProfile` Analysis:**
   - Unveils that while base triangular solves are accelerated by SciPy Level-2 BLAS, `dense_lu_factor` (base LU decomposition) remains unaccelerated in pure Python nested loops, and PFI eta-vector application runs sequentially in Python.

---

## 2. Profiling & Instrumentation Architecture

A dedicated, lightweight profiling subsystem was added in [`optiforge/lp/profiling.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/profiling.py):

### 2.1 Design Principles:
1. **Zero Runtime Overhead When Inactive:**
   - Default: `enable_profiling = False`.
   - All timing blocks are wrapped with `if prof_enabled:`. When disabled, zero `perf_counter` calls, zero allocations, and zero profiling logic execute.
2. **Lightweight Monotonic Timing:**
   - High-resolution timing via `time.perf_counter()`.
   - Granular tracking per solver phase:
     - `initial_factorization`: Initial basis factorization (DenseLU or PFI init).
     - `primal_solve`: Basic variable solution $B x_B = b$ (FTRAN 1).
     - `btran_y`: Dual multipliers $B^T y = c_B$ (BTRAN 1, skipped during incremental RC).
     - `gemv_aty`: Reduced cost evaluation $r = c - A^T y$.
     - `pricing`: Dual pricing / leaving row selection (infeasibility check & Bland tie-breaking).
     - `btran_v`: Dual direction vector $B^T v = e_p$ (BTRAN 2).
     - `gemv_atv`: Tableau row $\alpha = A^T v$.
     - `ratio_test`: Dual ratio test & entering variable selection.
     - `basis_update`: Basis array update and nonbasic bisect search.
     - `pfi_update`: Search direction $B d = A_{:, q}$ and PFI eta column replacement.
     - `rc_update`: Incremental reduced-cost & dual multiplier updates ($\Delta y, \Delta r$).
3. **Telemetry & Analytical Calculations:**
   - Component runtime share: $f_i = \frac{T_i}{T_{total}}$
   - Theoretical Amdahl maximum speedup: $S_{max\_i} = \frac{1}{1 - f_i}$
   - Call counts and average time per call.
   - Provider statistics integration (`get_provider_stats()`).

---

## 3. Empirical Benchmarks

### Table 1: End-to-End Runtime and Consistency Comparison

Workloads tested across dimensions using standard form Dual Revised Simplex PFI and Branch-and-Bound MILP:

| Workload | Dimensions ($m \times n$) | Iterations | Python Provider (ms) | SciPy Provider (ms) | Speedup | Objective Match | Iteration Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **30x60** | $30 \times 60$ | 5 | 2.14 ms | 1.32 ms | **1.63x** | EXACT ($< 10^{-12}$) | EXACT |
| **60x120** | $60 \times 120$ | 19 | 13.56 ms | 4.16 ms | **3.26x** | EXACT ($< 10^{-12}$) | EXACT |
| **100x200** | $100 \times 200$ | 14 | 15.52 ms | 4.74 ms | **3.27x** | EXACT ($< 10^{-12}$) | EXACT |
| **200x400** | $200 \times 400$ | 41 | 92.42 ms | 24.81 ms | **3.73x** | EXACT ($< 10^{-12}$) | EXACT |
| **MILP** | $1 \times 12$ knapsack | 49 nodes | 50.96 ms | 41.99 ms | **1.21x** | EXACT ($< 10^{-6}$) | EXACT |

---

### Table 2: Component Breakdown & Amdahl Maximums

#### Workload: $30 \times 60$ ($m=30, n=60$, 5 pivots)

| Component | Python Time (ms) | Python Share $f_i$ | Python $S_{max}$ | SciPy Time (ms) | SciPy Share $f_i$ | SciPy $S_{max}$ | Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `primal_solve` | 0.491 | 28.8% | 1.40x | 0.232 | 25.4% | 1.34x | 6 |
| `initial_factorization` | 0.253 | 14.8% | 1.17x | 0.226 | 24.8% | 1.33x | 1 |
| `pfi_update` | 0.392 | 23.0% | 1.30x | 0.169 | 18.5% | 1.23x | 5 |
| `btran_v` | 0.352 | 20.7% | 1.26x | 0.126 | 13.9% | 1.16x | 5 |
| `ratio_test` | 0.071 | 4.2% | 1.04x | 0.065 | 7.2% | 1.08x | 5 |
| `pricing` | 0.049 | 2.9% | 1.03x | 0.044 | 4.8% | 1.05x | 6 |
| `btran_y` | 0.068 | 4.0% | 1.04x | 0.021 | 2.3% | 1.02x | 1 |
| `rc_update` | 0.014 | 0.8% | 1.01x | 0.014 | 1.5% | 1.02x | 5 |
| `gemv_atv` | 0.007 | 0.4% | 1.00x | 0.007 | 0.8% | 1.01x | 5 |
| `basis_update` | 0.004 | 0.2% | 1.00x | 0.004 | 0.4% | 1.00x | 5 |
| `gemv_aty` | 0.003 | 0.2% | 1.00x | 0.003 | 0.3% | 1.00x | 1 |

*SciPy provider solves: 576, Fallbacks: 0.*

---

#### Workload: $60 \times 120$ ($m=60, n=120$, 19 pivots)

| Component | Python Time (ms) | Python Share $f_i$ | Python $S_{max}$ | SciPy Time (ms) | SciPy Share $f_i$ | SciPy $S_{max}$ | Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pfi_update` | 4.458 | 29.6% | 1.42x | 1.199 | 23.0% | 1.30x | 19 |
| `primal_solve` | 4.362 | 28.9% | 1.41x | 1.135 | 21.8% | 1.28x | 20 |
| `btran_v` | 3.458 | 22.9% | 1.30x | 1.047 | 20.1% | 1.25x | 19 |
| `initial_factorization` | 1.546 | 10.3% | 1.11x | 0.665 | 12.8% | 1.15x | 1 |
| `ratio_test` | 0.412 | 2.7% | 1.03x | 0.638 | 12.2% | 1.14x | 19 |
| `pricing` | 0.289 | 1.9% | 1.02x | 0.263 | 5.1% | 1.05x | 20 |
| `gemv_atv` | 0.067 | 0.4% | 1.00x | 0.118 | 2.3% | 1.02x | 19 |
| `rc_update` | 0.090 | 0.6% | 1.01x | 0.078 | 1.5% | 1.02x | 19 |
| `btran_y` | 0.342 | 2.3% | 1.02x | 0.032 | 0.6% | 1.01x | 1 |
| `basis_update` | 0.028 | 0.2% | 1.00x | 0.024 | 0.5% | 1.00x | 19 |
| `gemv_aty` | 0.020 | 0.1% | 1.00x | 0.007 | 0.1% | 1.00x | 1 |

*SciPy provider solves: 1080, Fallbacks: 0.*

---

#### Workload: $100 \times 200$ ($m=100, n=200$, 14 pivots)

| Component | Python Time (ms) | Python Share $f_i$ | Python $S_{max}$ | SciPy Time (ms) | SciPy Share $f_i$ | SciPy $S_{max}$ | Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `initial_factorization` | 1.581 | 12.3% | 1.14x | 1.203 | **33.3%** | **1.50x** | 1 |
| `primal_solve` | 3.906 | 30.4% | 1.44x | 0.771 | **21.3%** | **1.27x** | 15 |
| `pfi_update` | 3.573 | 27.8% | 1.39x | 0.607 | **16.8%** | **1.20x** | 14 |
| `btran_v` | 3.014 | 23.5% | 1.31x | 0.453 | **12.5%** | **1.14x** | 14 |
| `ratio_test` | 0.262 | 2.0% | 1.02x | 0.275 | **7.6%** | **1.08x** | 14 |
| `pricing` | 0.147 | 1.1% | 1.01x | 0.140 | **3.9%** | **1.04x** | 15 |
| `gemv_atv` | 0.074 | 0.6% | 1.01x | 0.063 | **1.8%** | **1.02x** | 14 |
| `rc_update` | 0.057 | 0.4% | 1.00x | 0.045 | **1.3%** | **1.01x** | 14 |
| `btran_y` | 0.210 | 1.6% | 1.02x | 0.031 | **0.8%** | **1.01x** | 1 |
| `basis_update` | 0.015 | 0.1% | 1.00x | 0.016 | **0.4%** | **1.00x** | 14 |
| `gemv_aty` | 0.011 | 0.1% | 1.00x | 0.011 | **0.3%** | **1.00x** | 1 |

*SciPy provider solves: 450, Fallbacks: 0.*

---

#### Workload: $200 \times 400$ ($m=200, n=400$, 41 pivots)

| Component | Python Time (ms) | Python Share $f_i$ | Python $S_{max}$ | SciPy Time (ms) | SciPy Share $f_i$ | SciPy $S_{max}$ | Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pfi_update` | 29.723 | 32.5% | 1.48x | 7.423 | **33.4%** | **1.50x** | 41 |
| `initial_factorization` | 8.515 | 9.3% | 1.10x | 4.697 | **21.1%** | **1.27x** | 1 |
| `primal_solve` | 23.954 | 26.2% | 1.35x | 4.568 | **20.5%** | **1.26x** | 42 |
| `btran_v` | 23.394 | 25.6% | 1.34x | 2.134 | **9.6%** | **1.11x** | 41 |
| `ratio_test` | 1.476 | 1.6% | 1.02x | 1.367 | **6.1%** | **1.07x** | 41 |
| `gemv_atv` | 1.048 | 1.1% | 1.01x | 0.919 | **4.1%** | **1.04x** | 41 |
| `pricing` | 0.606 | 0.7% | 1.01x | 0.539 | **2.4%** | **1.02x** | 42 |
| `rc_update` | 0.204 | 0.2% | 1.00x | 0.228 | **1.0%** | **1.01x** | 39 |
| `btran_y` | 2.300 | 2.5% | 1.03x | 0.199 | **0.9%** | **1.01x** | 3 |
| `gemv_aty` | 0.139 | 0.2% | 1.00x | 0.116 | **0.5%** | **1.00x** | 3 |
| `basis_update` | 0.085 | 0.1% | 1.00x | 0.067 | **0.3%** | **1.00x** | 41 |

*SciPy provider solves: 768, Fallbacks: 0.*

---

## 4. Statistical `cProfile` Analysis

On the $100 \times 200$ problem with 3 consecutive solves:

### Python Provider (`triangular_solve_provider="python"`):
- Total function calls: 33,142 in 0.058s.
- Top time-consuming functions:
  1. `back_substitution` ([`lu.py:370`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py#L370)): 0.016s cumulative (87 calls)
  2. `forward_substitution` ([`lu.py:324`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py#L324)): 0.012s cumulative (87 calls)
  3. `forward_substitution_transpose` ([`lu.py:411`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py#L411)): 0.010s cumulative (48 calls)
  4. `back_substitution_transpose` ([`lu.py:451`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py#L451)): 0.007s cumulative (48 calls)
  5. `dense_lu_factor` ([`lu.py:190`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py#L190)): 0.007s cumulative (3 calls)

### SciPy Provider (`triangular_solve_provider="scipy"`):
- Total function calls: 20,044 in 0.028s (~2.07x overall cProfile execution speedup).
- Top time-consuming functions:
  1. `solve_triangular` ([`scipy.linalg._basic.py:281`](file:///c:/Users/Prasanna/SIH2026/OptiForge/scratch/scipy_env/Lib/site-packages/scipy/linalg/_basic.py#L281)): 0.010s cumulative (270 calls)
  2. `dense_lu_factor` ([`lu.py:190`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py#L190)): **0.006s cumulative (3 calls)**
  3. `_raw_solve` & sequential eta evaluation ([`pivot_basis.py:333`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py#L333)): 0.008s cumulative (87 calls)

### Architectural Diagnosis:
- In SciPy mode, base triangular solve time has dropped substantially.
- **The base factorization `dense_lu_factor` (performed via pure Python Gaussian elimination) now accounts for a large fraction of setup time ($33.3\%$ on $100\times 200$, $21.1\%$ on $200\times 400$).**
- **The iterative update phase `pfi_update` (search direction FTRAN + eta vector incorporation) accounts for $33.4\%$ of total runtime on $200\times 400$.**

---

## 5. Measured Bottleneck Ranking After Stage 8.37

Under the accelerated SciPy architecture, the measured bottleneck ranking across mid-to-large workloads is:

| Rank | Component / Phase | Runtime Share ($f_i$) | Amdahl Upper Bound ($S_{max}$) | Architectural Cause |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **`pfi_update`** | **17% – 33%** | **1.20x – 1.50x** | Search direction column solve ($B d = A_{:, q}$) and sequential Python eta vector replacement. |
| **2** | **`initial_factorization`** | **21% – 33%** | **1.27x – 1.50x** | `dense_lu_factor` in [`lu.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py) runs pure Python loops for initial LU factorization without Level-3 LAPACK `getrf`/`lu_factor`. |
| **3** | **`primal_solve`** | **20% – 21%** | **1.25x – 1.27x** | Sequential forward/backward eta sweep in `PivotUpdateFactorization.solve(b)` combined with base solve. |
| **4** | **`btran_v`** | **10% – 13%** | **1.11x – 1.14x** | Backward transpose solve $B^T v = e_p$ (already benefiting from incremental reduced costs eliminating `btran_y`). |
| **5** | **`ratio_test`** | **6% – 8%** | **1.06x – 1.08x** | Ratio calculation and Bland tie-breaking. |
| **6** | **`gemv_atv`** | **2% – 4%** | **1.02x – 1.04x** | Tableau row calculation $A^T v$ via NumPy BLAS `dot`. |
| **7** | **`pricing`** | **2% – 5%** | **1.02x – 1.05x** | Basic infeasibility scan and row selection. |
| **8** | **`rc_update`** | **1.0% – 1.5%** | **1.01x – 1.02x** | Incremental update vectors ($y \mathrel{-}= \theta_D v, r \mathrel{+}= \theta_D \alpha$). |

---

## 6. Verification and Contract Preservation

1. **Numerical Invariance:**
   - Solutions, objective values, iterations, basis indices, and nonbasic indices are **identical** between `enable_profiling=False` and `enable_profiling=True`.
2. **Provider Contract:**
   - `triangular_solve_provider="python"` remains the default, 100% dependency-free NumPy implementation.
   - `triangular_solve_provider="scipy"` remains strictly optional and opt-in.
3. **Repository Regression:**
   - **1040 tests passed**, 0 failed, 19 benign warnings.
4. **Constraint Enforcement:**
   - No numerical tolerances were modified.
   - No pivot rules, ratio tests, or pricing heuristics were modified.
   - No PFI mathematics or Compact WY algorithms were modified.
   - No premature optimizations were applied.

---

## 7. Recommendation for Next Optimization Stage

The empirical data gathered in Stage 8.38 isolates two candidate bottlenecks under the SciPy provider architecture:
1. **Candidate A: Initial Base LU Factorization (`dense_lu_factor`):**
   - Accounts for **21% to 33%** of solve time on $100\times 200$ and $200\times 400$.
   - Currently, `lu.py` uses pure Python Gaussian elimination with partial pivoting. Extending the controlled provider abstraction to `lu_factor` (Level-3 BLAS LAPACK `dgetrf`) would immediately accelerate initial and periodic refactorizations.
2. **Candidate B: PFI Column Update & Eta Application (`pfi_update`):**
   - Accounts for **17% to 33%** of solve time.
   - Vectorizing eta matrix applications or Compact WY representation for multiple rank-1 updates.

Candidate A represents the most direct, low-risk continuation of the controlled provider architecture introduced in Stage 8.37.
