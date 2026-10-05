# Stage 8.7 — Mathematical Contract: Product-Form Incremental Basis Update

## 1. Mathematical Formulation

Let $B_0 \in \mathbb{R}^{m \times m}$ be a non-singular basis matrix with dense LU factorization:
$$P_0 B_0 = L_0 U_0$$
where $P_0$ is a permutation matrix, $L_0$ is unit lower triangular, and $U_0$ is upper triangular.

### 1.1 Simplex Column Replacement
In simplex iteration $k \ge 1$, column $p \in \{0, \dots, m-1\}$ of basis $B_{k-1}$ is replaced by a new column vector $a_q \in \mathbb{R}^m$:
$$B_k = B_{k-1} + (a_q - B_{k-1} e_p) e_p^T$$
where $e_p$ is the $p$-th standard basis vector.

Using the simplex search direction vector $d = B_{k-1}^{-1} a_q$:
$$B_k = B_{k-1} (I + (d - e_p) e_p^T) = B_{k-1} E_k$$
where $E_k$ is an elementary eta matrix having column $p$ equal to $d$ and column $j$ equal to $e_j$ for $j \ne p$.

### 1.2 Inverse Representation (Product Form of Inverse)
The inverse of $E_k$ is an elementary eta matrix of the exact same sparsity pattern:
$$E_k^{-1} = I + (\eta^{(k)} - e_p) e_p^T$$
where the eta vector $\eta^{(k)} \in \mathbb{R}^m$ is defined by pivot element $\delta = d_p \ne 0$:
$$\eta^{(k)}_i = \begin{cases} - \frac{d_i}{\delta} & \text{if } i \ne p \\ \frac{1}{\delta} & \text{if } i = p \end{cases}$$

After $k$ updates following the base factorization $B_0$:
$$B_k = B_0 E_1 E_2 \cdots E_k$$
$$B_k^{-1} = E_k^{-1} E_{k-1}^{-1} \cdots E_1^{-1} B_0^{-1}$$

### 1.3 Primal Solve ($B_k x = b$)
To solve $B_k x = b$:
$$x = B_k^{-1} b = E_k^{-1} \left( E_{k-1}^{-1} \dots \left( E_1^{-1} (B_0^{-1} b) \right) \dots \right)$$

1. Compute $v_0 = B_0^{-1} b$ using base DenseLU factors ($P_0 B_0 = L_0 U_0$) via forward/back substitution ($O(m^2)$ flops).
2. For $\ell = 1, \dots, k$ sequentially (chronological order):
   - With $p = p_\ell$ and $\eta = \eta^{(\ell)}$:
   - $\alpha = v_{\ell-1}[p]$
   - $v_\ell[i] = v_{\ell-1}[i] + \alpha \cdot \eta[i]$ for $i \ne p$
   - $v_\ell[p] = \alpha \cdot \eta[p]$
   - Flops per eta: $2m$ flops.

### 1.4 Transpose / Dual Solve ($B_k^T y = c$)
To solve $B_k^T y = c$:
$$B_k^{-T} = (E_k^{-1} \cdots E_1^{-1} B_0^{-1})^T = B_0^{-T} E_1^{-T} \cdots E_k^{-T}$$
$$y = B_k^{-T} c = B_0^{-T} \left( E_1^{-T} \left( E_2^{-T} \dots \left( E_k^{-T} c \right) \dots \right) \right)$$

1. Set $u_k = c$.
2. For $\ell = k, k-1, \dots, 1$ in reverse chronological order:
   - With $p = p_\ell$ and $\eta = \eta^{(\ell)}$:
   - $u_{\ell-1}[i] = u_\ell[i]$ for $i \ne p$
   - $u_{\ell-1}[p] = \eta^T u_\ell = \sum_{i=0}^{m-1} \eta[i] \cdot u_\ell[i]$
   - Flops per eta: $2m$ flops (one dot product).
3. Compute $y = B_0^{-T} u_0$ using base DenseLU factors via transpose triangular solves ($O(m^2)$ flops).

## 2. Invariant & Degradation Contract

1. **Exact Mathematical Equivalence**: In exact arithmetic, $B_k x_{\text{inc}} = b$ and $B_k^T y_{\text{inc}} = c$ match $B_k^{-1} b$ and $B_k^{-T} c$ identically.
2. **Error Accumulation**: Each eta application introduces $O(\epsilon_{\text{mach}})$ floating-point roundoff. Dividing by small pivot $\delta$ amplifies incoming perturbations by $|1/\delta|$.
3. **Refactorization Policy**:
   Full refactorization ($P B_k = L U$) is triggered whenever:
   - Update count exceeds $N_{\text{max}}$ (default: 30 updates).
   - Pivot magnitude $|\delta| < \text{pivot\_tolerance}$ (singular/near-singular pivot).
   - Relative residual $r_{\text{rel}} = \frac{\|B_k x - b\|_\infty}{\|B_k\|_\infty \|x\|_\infty + \|b\|_\infty} > \tau_{\text{refactor}}$ (default: $10^{-8}$).
   - Any NaN, Inf, or numerical singularity is encountered.
4. **Fallback Contract**:
   If an incremental solve yields unacceptable residual or an update is rejected, the incremental solver immediately refactorizes the current explicit basis matrix $B$ using reference `dense_lu_factor`, resets all etas, and re-solves.
