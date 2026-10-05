# Stage 8.8 — Mathematical Contract: Dual Revised Simplex

## 1. Mathematical Formulation

Consider a Linear Program in standard form:
$$\min c^T x \quad \text{subject to } A x = b, \; x \ge 0$$
(or $\max c^T x$ subject to $A x = b, \; x \ge 0$), with $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, $c \in \mathbb{R}^n$.

Let $\mathcal{B} = (j_0, j_1, \dots, j_{m-1})$ be an ordered set of basic column indices, and $\mathcal{N}$ be the nonbasic column indices.
The basis matrix is $B = A[:, \mathcal{B}] \in \mathbb{R}^{m \times m}$.

### 1.1 Invariants & Solutions
1. **Primal Basic Solution**:
   $$x_B = B^{-1} b, \quad x_N = 0$$
   Primal feasibility requires $x_B[i] \ge -\epsilon_{\text{feas}}$ for all $i \in \{0, \dots, m-1\}$.
2. **Dual Basic Solution**:
   $$B^T y = c_B \implies y = B^{-T} c_B$$
3. **Reduced Costs**:
   $$r = c - A^T y \in \mathbb{R}^n$$
   By definition, $r_B = c_B - B^T y = 0$.
   For nonbasic variables $j \in \mathcal{N}$:
   - **For MINIMIZE**: Dual feasibility requires $r_j \ge -\epsilon_{\text{rc}}$ for all $j \in \mathcal{N}$.
   - **For MAXIMIZE**: Dual feasibility requires $r_j \le \epsilon_{\text{rc}}$ for all $j \in \mathcal{N}$.

### 1.2 Dual Simplex Starting Contract
The Dual Revised Simplex algorithm requires:
1. **Dual Feasibility**: satisfied at the initial basis.
2. **Primal Infeasibility**: $x_B[i] < -\epsilon_{\text{feas}}$ for at least one basic row $i$.

If $x_B \ge -\epsilon_{\text{feas}}$ initially, the basis is already primal feasible and dual feasible, hence **optimal**.

---

## 2. Pivot Mechanism

### 2.1 Leaving Row Selection (Dual Pricing)
Identify all primal infeasible rows:
$$\mathcal{P}_{\text{infeas}} = \{ i \in \{0, \dots, m-1\} : x_B[i] < -\epsilon_{\text{feas}} \}$$

If $\mathcal{P}_{\text{infeas}} = \emptyset$, terminate: **OPTIMAL**.

Otherwise, choose leaving row $p$:
- **Most Infeasible (Dantzig rule)**: $p = \arg\min_{i} x_B[i]$.
- **Tie-breaking (Bland's rule)**: Among rows with $x_B[i] \approx \min_k x_B[k]$, choose row $p$ with smallest basic variable index $j_{\text{leave}} = \mathcal{B}[p]$.

### 2.2 Dual Search Direction (BTRAN)
To determine how reduced costs will change when row $p$ is pivoted, compute the $p$-th row of the basis inverse:
$$B^T v = e_p \implies v = B^{-T} e_p$$
where $e_p \in \mathbb{R}^m$ is the $p$-th unit vector.

Compute the transformed tableau row across all columns:
$$\alpha_j = A_j^T v = A[:, j]^T v \quad \text{for } j \in \{0, \dots, n-1\}$$
Properties:
- For basic variables: $\alpha_{\mathcal{B}[i]} = e_p^T B^{-1} B e_i = \delta_{pi}$. Thus $\alpha_{\mathcal{B}[p]} = 1$, and $\alpha_{\mathcal{B}[i]} = 0$ for $i \ne p$.

### 2.3 Dual Ratio Test (Entering Column Selection)
When row $p$ leaves the basis, its value increases toward 0 by updating $y_{\text{new}} = y + \theta v$ (for appropriate sign).
To maintain dual feasibility:
- Eligible entering nonbasic columns are those with negative tableau coefficient:
  $$\mathcal{E} = \{ j \in \mathcal{N} : \alpha_j < -\epsilon_{\text{pivot}} \}$$

**Infeasibility Condition**:
If $\mathcal{E} = \emptyset$, then all $\alpha_j \ge -\epsilon_{\text{pivot}}$ for $j \in \mathcal{N}$.
In the primal, row $p$ asserts:
$$x_B[p] + \sum_{j \in \mathcal{N}} \alpha_j x_j = b_p < 0$$
Since $x_j \ge 0$ and all $\alpha_j \ge 0$, the left-hand side is non-negative while the RHS is strictly negative.
No feasible solution exists.
**Termination**: **INFEASIBLE**.

**Ratio Calculation**:
Define adjusted reduced cost $\bar{r}_j$:
$$\bar{r}_j = \begin{cases} r_j & \text{if MINIMIZE} \\ -r_j & \text{if MAXIMIZE} \end{cases}$$
Dual feasibility guarantees $\bar{r}_j \ge -\epsilon_{\text{rc}}$. Clamp tiny negative values $\bar{r}_j = \max(0, \bar{r}_j)$.
For each $j \in \mathcal{E}$, the maximum step preserving dual feasibility is:
$$\theta_j = \frac{\bar{r}_j}{-\alpha_j} \ge 0$$

Let $\theta^* = \min_{j \in \mathcal{E}} \theta_j$.

**Entering Column Tie-Breaking (Bland's Rule)**:
Among candidates with $|\theta_j - \theta^*| \le 10^{-11}$, select $q = j_{\text{enter}} = \min \{ j \in \mathcal{E}_{\text{min}} \}$.

### 2.4 Basis Update & Factorization
- Leaving variable $j_{\text{leave}} = \mathcal{B}[p]$.
- Entering variable $j_{\text{enter}} = q$.
- Basis update: $\mathcal{B}[p] \leftarrow q$.
- Nonbasic update: $\mathcal{N} \leftarrow (\mathcal{N} \setminus \{q\}) \cup \{j_{\text{leave}}\}$, sorted.
- Solve mechanism:
  - If using `DenseLU`: compute $P B_{\text{new}} = L U$.
  - If using `IncrementalBasisFactorization`: update column $p$ with $A[:, q]$ and search direction $d = B^{-1} A[:, q]$ where $d[p] = \alpha_q < 0 \ne 0$.

---

## 3. Numerical Safeguards & Fallbacks

1. **Singularity Guard**: If $|\alpha_q| < \epsilon_{\text{pivot}}$, pivot is rejected and full refactorization is performed.
2. **Residual Verification**: Compute $\|B x_B - b\|_\infty / (\|B\|_\infty \|x_B\|_\infty + \|b\|_\infty)$ at termination and optionally during pivots.
3. **Iteration Limit**: Terminate with `ITERATION_LIMIT` if iterations exceed `max_iterations`.
