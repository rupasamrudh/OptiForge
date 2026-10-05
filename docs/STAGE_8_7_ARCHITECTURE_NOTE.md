# Stage 8.7 — Architecture Note: Incremental Basis Update & Solver Lifecycle

## 1. Inspection of Existing Solver Architecture

### 1.1 Simplex Iteration Lifecycle
In `optiforge/lp/simplex.py` (`_simplex_loop`), each iteration follows:
1. Basis validation: `len(basis) == m`, uniqueness of indices.
2. Basis extraction & factorization:
   - Extract $B = A[:, \text{basis}]$ from scratch.
   - Recompute full dense LU with partial pivoting: `basis_fact = BasisFactorization(B, pivot_tolerance)`.
   - Flop complexity: $\frac{2}{3} m^3 + O(m^2)$.
   - Solve primal basic values: $x_B = B^{-1} b$ via forward/back substitution ($O(m^2)$).
3. Dual solve:
   - Extract $c_B = c[\text{basis}]$.
   - Solve $B^T y = c_B$ via transpose forward/back substitution ($O(m^2)$).
4. Pricing (Reduced Costs):
   - Vectorized GEMV: $A^T y$.
   - $r_{\text{all}} = c - A^T y$.
   - Identify candidate entering variables ($r_j < -\text{tol}$ for MIN, $r_j > \text{tol}$ for MAX).
5. Optimality check:
   - If no candidates, terminate with `OPTIMAL`.
6. Ratio test:
   - Select entering variable $j_{\text{enter}}$ via Bland's rule (smallest column index).
   - Compute search direction: solve $B d = A[:, j_{\text{enter}}]$ ($O(m^2)$).
   - If all $d_i \le \text{tol}$, terminate with `UNBOUNDED`.
   - Minimum ratio test with Bland's rule tie-breaking selects leaving row $i_{\text{leave}}$ and leaving column $j_{\text{leave}} = \text{basis}[i_{\text{leave}}]$.
7. Basis update:
   - `basis[i_leave] = j_enter`.
   - `nonbasic.remove(j_enter); nonbasic.append(leaving_var); nonbasic.sort()`.
   - Cycle repeats at Step 1, discarding the previous factorization entirely.

### 1.2 Identified Redundancies & Opportunities
- At iteration $k$, the basis matrix $B_{k+1}$ differs from $B_k$ by exactly ONE column at index $p = i_{\text{leave}}$:
  $$B_{k+1} = B_k + (a_{j_{\text{enter}}} - B_k e_p) e_p^T$$
- The search direction $d = B_k^{-1} a_{j_{\text{enter}}}$ is ALREADY COMPUTED during Step 6 of iteration $k$.
- In the existing implementation, $B_{k+1}$ is factored from scratch at $O(m^3)$ cost, even though $B_{k+1}^{-1} = E_k^{-1} B_k^{-1}$ where $E_k^{-1}$ is an elementary eta matrix computable in $O(m)$ flops from $d$ and $p$.
- Over an LP with 100 pivots on a $50 \times 50$ system, the current solver performs 100 full $O(m^3)$ factorizations. An incremental update scheme would perform 1 initial factorization and up to 100 fast $O(m)$ updates, with periodic refactorizations only when numerical error dictates.

### 1.3 Stability & Monitor Interaction
- `NumericalStabilityMonitor` currently logs LU factorization diagnostics, pivot magnitudes, residuals, condition estimates, and failures.
- When an incremental update scheme is active, the monitor can additionally track `NumericalEventType.REFACTORIZATION` events and condition/residual degradation.
- Singularity is detectable at pivot time: $|d[i_{\text{leave}}| < \text{pivot\_tolerance}$ indicates numerical singularity or breakdown.
- Immediate safe fallback to full `DenseLU` is mandatory if residuals exceed tolerances or numerical issues arise.
