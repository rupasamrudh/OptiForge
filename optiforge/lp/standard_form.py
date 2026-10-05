"""LP standard-form conversion and Phase-I preparation.

Transforms a general LP model into the standard form required by the
Revised Simplex method:

    max / min   c^T x
    subject to  A x  = b
                x   >= 0

Conversion rules
----------------
1.  **Negative RHS Normalization**:
    Every constraint is normalized so that b_i >= 0.
    If b_i < 0, the constraint is multiplied by -1 and its sense is flipped:
        a^T x <= -4   -->   -a^T x >= 4
        a^T x >= -4   -->   -a^T x <= 4
        a^T x == -4   -->   -a^T x == 4

2.  **Slack, Surplus, and Artificial Variables**:
    *   ``<=`` constraint  -->  add a **slack** variable s >= 0:
            a^T x <= b   becomes   a^T x + s = b
            s provides an initial basic column (+1 in row i).

    *   ``>=`` constraint  -->  add a **surplus** variable e >= 0 and an
        **artificial** variable a >= 0:
            a^T x >= b   becomes   a^T x - e + a = b
            a provides an initial basic column (+1 in row i) for Phase I.

    *   ``==`` constraint  -->  add an **artificial** variable a >= 0:
            a^T x == b   becomes   a^T x + a = b
            a provides an initial basic column (+1 in row i) for Phase I.

3.  **Complete Phase-I Basis Invariant**:
    For an m-row system, the initial Phase-I basis contains exactly m
    linearly independent columns (one slack or artificial variable per row),
    forming an m x m identity basis matrix B = I_m.

4.  **Phase-I Objective**:
    Minimizes w = sum(a_i) for all artificial variables.
    phase1_c[artificial_idx] = 1.0, and 0.0 for all other columns.
    The original objective c is preserved independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from optiforge.model.constraint import ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import ObjectiveSense
from optiforge.model.variable import VariableType


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.exceptions import (
    NumericalSingularityError,
    SingularBasisError,
)


# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------

class VariableRole(Enum):
    """Role of a variable in the standard-form model."""

    ORIGINAL = "original"
    SLACK = "slack"            # added for <= constraints (+1 coefficient)
    SURPLUS = "surplus"        # added for >= constraints (-1 coefficient)
    ARTIFICIAL = "artificial"  # added for >= and == constraints (+1 coefficient, Phase I)


@dataclass
class VariableInfo:
    """Metadata for a single column in the standard-form A matrix.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. ``x1``, ``s1``, ``e2``, ``a2``).
    role : VariableRole
        Whether this column represents an original, slack, surplus,
        or artificial variable.
    column_index : int
        Zero-based position in the A matrix / c vector.
    original_lower_bound : float
        Lower bound from the original model (relevant for original variables).
    original_upper_bound : float
        Upper bound from the original model (relevant for original variables).
    """

    name: str
    role: VariableRole
    column_index: int
    original_lower_bound: float = 0.0
    original_upper_bound: float = math.inf


# ---------------------------------------------------------------------------
# Standard-form model
# ---------------------------------------------------------------------------

@dataclass
class StandardFormModel:
    """LP in standard form with complete Phase-I preparation.

    Standard form:
        minimize / maximize   c^T x
        subject to            A x = b
                              x  >= 0

    Phase-I auxiliary problem:
        minimize              phase1_c^T x = sum(a_i)
        subject to            A x = b
                              x  >= 0

    Attributes
    ----------
    A : np.ndarray
        Constraint matrix (m x n), dense float64.
    b : np.ndarray
        Right-hand-side vector (m,), guaranteed non-negative (b >= 0).
    c : np.ndarray
        Original objective coefficient vector (n,).
    phase1_c : np.ndarray
        Phase-I objective coefficient vector (n,) with 1.0 on artificial
        variables and 0.0 elsewhere.
    variable_names : list[str]
        Ordered column names matching A, c, and phase1_c.
    variable_info : list[VariableInfo]
        Per-column metadata (role, bounds, index).
    original_variable_mapping : dict[str, int]
        Original variable name -> column index in A.
    slack_indices : list[int]
        Column indices corresponding to slack variables.
    surplus_indices : list[int]
        Column indices corresponding to surplus variables.
    artificial_indices : list[int]
        Column indices corresponding to artificial variables.
    phase1_basis_indices : list[int]
        Ordered list of m column indices forming the initial Phase-I basis.
    phase1_nonbasic_indices : list[int]
        Column indices not in phase1_basis_indices.
    basis_indices : list[int]
        Current basis column indices (equals phase1_basis_indices at initialization).
    nonbasic_indices : list[int]
        Current nonbasic column indices.
    objective_sense : ObjectiveSense
        MAXIMIZE or MINIMIZE — preserved from the original model.
    phase1_objective_sense : ObjectiveSense
        Always MINIMIZE for Phase I.
    num_original_variables : int
        Number of original decision variables.
    num_slack_variables : int
        Number of slack variables generated.
    num_surplus_variables : int
        Number of surplus variables generated.
    num_artificial_variables : int
        Number of artificial variables generated.
    requires_artificial : list[int]
        Row indices (0-based) that require artificial variables.
    upper_bounds : dict[int, float]
        Column index -> finite upper bound for bounded-variable tracking.
    variable_shifts : dict[int, float]
        Column index -> lower-bound shift L for original variables with
        non-zero lower bounds (x = y + L, y >= 0).
    objective_constant : float
        Constant shift accumulated from variable lower-bound shifts:
        original_obj = c^T y + objective_constant.
    """

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    phase1_c: np.ndarray
    variable_names: list[str]
    variable_info: list[VariableInfo]
    original_variable_mapping: dict[str, int]
    slack_indices: list[int]
    surplus_indices: list[int]
    artificial_indices: list[int]
    phase1_basis_indices: list[int]
    phase1_nonbasic_indices: list[int]
    basis_indices: list[int]
    nonbasic_indices: list[int]
    objective_sense: ObjectiveSense
    num_original_variables: int
    num_slack_variables: int
    num_surplus_variables: int
    num_artificial_variables: int
    requires_artificial: list[int] = field(default_factory=list)
    upper_bounds: dict[int, float] = field(default_factory=dict)
    variable_shifts: dict[int, float] = field(default_factory=dict)
    objective_constant: float = 0.0
    phase1_objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE

    # -- derived properties ---------------------------------------------------

    @property
    def num_variables(self) -> int:
        """Total number of columns (original + slack + surplus + artificial)."""
        return self.A.shape[1]

    @property
    def num_constraints(self) -> int:
        """Number of rows (equality constraints)."""
        return self.A.shape[0]

    @property
    def needs_phase1(self) -> bool:
        """Return True if artificial variables were added and Phase I is needed."""
        return len(self.artificial_indices) > 0

    @property
    def has_upper_bounds(self) -> bool:
        """Return True if any variable has a finite upper bound."""
        return len(self.upper_bounds) > 0

    # -- basis validation -----------------------------------------------------

    def validate_basis(self, basis: list[int] | None = None) -> None:
        """Validate structural properties of a basis candidate.

        Parameters
        ----------
        basis : list[int] | None
            Column indices to validate. If None, validates `basis_indices`.

        Raises
        ------
        ValueError
            If basis length != m, indices are out of bounds, or indices are
            not unique.
        """
        if basis is None:
            basis = self.basis_indices

        m = self.num_constraints
        n = self.num_variables

        if len(basis) != m:
            raise ValueError(
                f"Basis has {len(basis)} columns but problem has {m} "
                "constraints — basis must have exactly m columns."
            )

        if len(set(basis)) != m:
            raise ValueError(
                f"Basis indices must be unique. Provided indices: {basis}"
            )

        for col_idx in basis:
            if col_idx < 0 or col_idx >= n:
                raise ValueError(
                    f"Basis index {col_idx} is out of bounds for {n} columns."
                )

        B = self.A[:, basis]
        if B.shape != (m, m):
            raise ValueError(
                f"Basis matrix shape is {B.shape}, expected ({m}, {m})."
            )

    # -- BFS computation ------------------------------------------------------

    def compute_initial_bfs(
        self, tol: float = 1e-10,
    ) -> dict[str, float]:
        """Compute the initial basic feasible solution (BFS) for Phase II.

        This method is valid when no artificial variables are present
        (i.e. `needs_phase1 == False`).

        Parameters
        ----------
        tol : float
            Numerical tolerance for non-negativity checks.

        Returns
        -------
        dict[str, float]
            Variable name -> value in the initial BFS.

        Raises
        ------
        ValueError
            If the model requires Phase I (artificial variables are present).
        SingularBasisError
            If the basis matrix B is singular.
        """
        if self.needs_phase1:
            raise ValueError(
                "Cannot compute initial BFS directly: constraint row(s) "
                f"{self.requires_artificial} require artificial variables. "
                "Use compute_phase1_bfs() for Phase-I initialization."
            )

        return self._solve_basis_system(self.basis_indices, tol=tol)

    def compute_phase1_bfs(
        self, tol: float = 1e-10,
    ) -> dict[str, float]:
        """Compute the initial basic feasible solution for Phase I.

        Solves B x_B = b using `phase1_basis_indices`. For the initial
        slack/artificial basis, B = I_m and x_B = b >= 0.

        Parameters
        ----------
        tol : float
            Numerical tolerance for non-negativity checks.

        Returns
        -------
        dict[str, float]
            Variable name -> value in the Phase-I BFS.

        Raises
        ------
        SingularBasisError
            If the basis matrix B is singular.
        ValueError
            If the computed solution violates x >= 0.
        """
        return self._solve_basis_system(self.phase1_basis_indices, tol=tol)

    def _solve_basis_system(
        self, basis: list[int], tol: float = 1e-10,
    ) -> dict[str, float]:
        """Internal solver for B x_B = b."""
        self.validate_basis(basis)
        m = self.num_constraints

        B = self.A[:, basis]
        try:
            basis_fact = BasisFactorization(B, pivot_tolerance=tol)
            x_B = basis_fact.solve(self.b)
        except SingularBasisError as err:
            raise SingularBasisError(
                f"Basis matrix B is singular or rank-deficient: {err}"
            ) from err

        # Validate non-negativity of basic variables.
        if np.any(x_B < -tol):
            neg_vars = [
                (self.variable_names[basis[i]], float(x_B[i]))
                for i in range(m)
                if x_B[i] < -tol
            ]
            raise ValueError(
                "Initial basic solution is infeasible — negative basic "
                f"variables: {neg_vars}"
            )

        # Build full variable-value mapping (nonbasic = 0).
        values: dict[str, float] = {name: 0.0 for name in self.variable_names}
        for k, col_idx in enumerate(basis):
            # Guard against floating-point -0.0
            val = float(x_B[k])
            values[self.variable_names[col_idx]] = 0.0 if abs(val) < tol else val

        return values

    # -- objective evaluation -------------------------------------------------

    def compute_objective_value(
        self, variable_values: dict[str, float],
    ) -> float:
        """Evaluate original objective c^T x for variable assignments."""
        x = np.array(
            [variable_values.get(name, 0.0) for name in self.variable_names],
            dtype=np.float64,
        )
        return float(self.c @ x)

    def compute_phase1_objective_value(
        self, variable_values: dict[str, float],
    ) -> float:
        """Evaluate Phase-I auxiliary objective sum(a_i)."""
        x = np.array(
            [variable_values.get(name, 0.0) for name in self.variable_names],
            dtype=np.float64,
        )
        return float(self.phase1_c @ x)

    # -- pretty printing ------------------------------------------------------

    def summary(self) -> str:
        """Return a multi-line textual summary of the standard form."""
        lines = [
            "Standard-Form LP (Phase-I Ready)",
            f"  Objective sense        : {self.objective_sense.value}",
            f"  Variables (total)      : {self.num_variables}",
            f"    original             : {self.num_original_variables}",
            f"    slack                : {self.num_slack_variables}",
            f"    surplus              : {self.num_surplus_variables}",
            f"    artificial           : {self.num_artificial_variables}",
            f"  Constraints            : {self.num_constraints}",
            f"  Phase-I basis columns  : {self.phase1_basis_indices}",
            f"  Phase-I nonbasic cols  : {self.phase1_nonbasic_indices}",
            f"  Needs Phase I          : {self.needs_phase1}",
        ]
        if self.requires_artificial:
            lines.append(
                f"  Artificial rows        : {self.requires_artificial}"
            )
        if self.upper_bounds:
            lines.append(f"  Finite upper bds       : {self.upper_bounds}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversion function
# ---------------------------------------------------------------------------

def to_standard_form(model: Model) -> StandardFormModel:
    """Convert a user-facing :class:`Model` to a Phase-I ready standard form.

    Applies negative RHS normalization (b_i >= 0), adds necessary slack,
    surplus, and artificial variables, constructs an explicit Phase-I
    objective vector, and guarantees a complete m-element initial basis.

    Bounds handling
    ---------------
    *   **Non-zero lower bounds** are handled by a variable shift:
        ``x = y + L`` with ``y >= 0``.  The objective gains a constant
        ``c_j * L_j`` and every constraint RHS is reduced by ``a_ij * L_j``.
        The shift is recorded in ``variable_shifts`` so postsolve can
        reconstruct ``x = y + L``.

    *   **Finite upper bounds** are represented as explicit ``<=`` rows
        (``x <= U`` becomes ``x + s = U`` with a slack variable).  This keeps
        the Revised Simplex pivot/ratio logic unchanged and never silently
        converts a bounded variable into an unbounded one.

    Parameters
    ----------
    model : Model
        A fully-specified LP (variables, constraints, objective).

    Returns
    -------
    StandardFormModel
        The equivalent LP in standard form (Ax = b, x >= 0) with Phase-I
        preparation.

    Raises
    ------
    ValueError
        If the model is incomplete, contains integer/binary variables,
        or contains free variables (lb = -inf).
    """
    # ── validate model completeness ──────────────────────────────────
    if model.num_variables == 0:
        raise ValueError("Model has no variables.")
    if model.objective is None:
        raise ValueError("Model must have an objective before conversion.")
    # Allow models with no explicit constraints if they have finite upper bounds
    # (which will be converted to constraints). Pure unbounded models need constraints.
    has_finite_upper_bounds = any(
        math.isfinite(v.upper_bound) for v in model.variables.values()
    )
    if model.num_constraints == 0 and not has_finite_upper_bounds:
        raise ValueError("Model has no constraints and no finite upper bounds.")

    # ── ordered original variables ───────────────────────────────────
    orig_vars = list(model.variables.values())
    n_orig = len(orig_vars)

    # ── validate variable bounds & types ─────────────────────────────
    upper_bounds: dict[int, float] = {}
    variable_shifts: dict[int, float] = {}
    objective_constant = 0.0
    for i, var in enumerate(orig_vars):
        if var.variable_type is not VariableType.CONTINUOUS:
            type_str = var.variable_type.value if hasattr(var.variable_type, "value") else str(var.variable_type)
            raise ValueError(
                f"Variable '{var.name}' is {type_str}. "
                "Only continuous variables are supported for LP standard-form "
                "conversion."
            )
        if var.lower_bound == -math.inf:
            raise ValueError(
                f"Free variable '{var.name}' (lb = -∞) is not yet "
                "supported in standard-form conversion."
            )
        if var.lower_bound != 0.0:
            # Variable shift: x = y + L with y >= 0.
            variable_shifts[i] = float(var.lower_bound)
            objective_constant += float(model.objective.coefficients.get(var.name, 0.0)) * float(var.lower_bound)
        if math.isfinite(var.upper_bound):
            upper_bounds[i] = var.upper_bound

    # ── constraint normalization (b_i >= 0) ──────────────────────────
    # If rhs < 0, multiply row by -1 and flip sense before column allocation.
    # Variable shifts (x = y + L) are applied FIRST, then RHS is normalized.
    constraints = list(model.constraints.values())
    m = len(constraints)

    var_index: dict[str, int] = {v.name: i for i, v in enumerate(orig_vars)}

    normalized_constrs = []
    for con in constraints:
        rhs = float(con.rhs)
        coeffs = dict(con.coefficients)
        sense = con.sense

        # Apply variable shifts FIRST: x_j = y_j + L_j  =>  rhs -= a_j * L_j
        for var_name, coeff in coeffs.items():
            j = var_index[var_name]
            if j in variable_shifts:
                rhs -= coeff * variable_shifts[j]

        # THEN normalize RHS to be non-negative
        if rhs < 0.0:
            rhs = -rhs
            coeffs = {var_name: -val for var_name, val in coeffs.items()}
            if sense is ConstraintSense.LESS_EQUAL:
                sense = ConstraintSense.GREATER_EQUAL
            elif sense is ConstraintSense.GREATER_EQUAL:
                sense = ConstraintSense.LESS_EQUAL
            # EQUAL remains EQUAL

        normalized_constrs.append({
            "name": con.name,
            "coefficients": coeffs,
            "sense": sense,
            "rhs": rhs,
        })

    # ── upper-bound rows: y_i <= U_i - L_i (explicit <= rows with slack) ──
    # Each finite upper bound becomes a dedicated row so the Revised Simplex
    # pivot/ratio logic is unchanged and bounded variables are never silently
    # converted into unbounded ones.
    upper_bound_rows: list[tuple[int, float]] = []
    for i, var in enumerate(orig_vars):
        if math.isfinite(var.upper_bound):
            bound = float(var.upper_bound) - variable_shifts.get(i, 0.0)
            upper_bound_rows.append((i, bound))

    m_total = m + len(upper_bound_rows)

    # ── count columns to allocate ────────────────────────────────────
    n_slack = 0
    n_surplus = 0
    n_artificial = 0

    for norm in normalized_constrs:
        sense = norm["sense"]
        if sense is ConstraintSense.LESS_EQUAL:
            n_slack += 1
        elif sense is ConstraintSense.GREATER_EQUAL:
            n_surplus += 1
            n_artificial += 1
        else:  # EQUAL
            n_artificial += 1

    n_slack += len(upper_bound_rows)  # one slack per upper-bound row
    n_total = n_orig + n_slack + n_surplus + n_artificial

    # ── allocate matrices and vectors ────────────────────────────────
    A = np.zeros((m_total, n_total), dtype=np.float64)
    b = np.zeros(m_total, dtype=np.float64)
    c = np.zeros(n_total, dtype=np.float64)
    phase1_c = np.zeros(n_total, dtype=np.float64)

    # ── variable bookkeeping ─────────────────────────────────────────
    variable_names: list[str] = [v.name for v in orig_vars]
    variable_info: list[VariableInfo] = [
        VariableInfo(
            name=v.name,
            role=VariableRole.ORIGINAL,
            column_index=i,
            original_lower_bound=v.lower_bound,
            original_upper_bound=v.upper_bound,
        )
        for i, v in enumerate(orig_vars)
    ]
    original_variable_mapping: dict[str, int] = {
        v.name: i for i, v in enumerate(orig_vars)
    }

    # ── fill original objective vector c ─────────────────────────────
    for var_name, coeff in model.objective.coefficients.items():
        c[original_variable_mapping[var_name]] = coeff

    # ── populate A, b, and generated columns ─────────────────────────
    extra_col = n_orig
    slack_indices: list[int] = []
    surplus_indices: list[int] = []
    artificial_indices: list[int] = []
    phase1_basis_indices: list[int] = []
    requires_artificial: list[int] = []

    for row, norm in enumerate(normalized_constrs):
        # Fill original variable coefficients in row
        for var_name, coeff in norm["coefficients"].items():
            A[row, original_variable_mapping[var_name]] = coeff

        b[row] = norm["rhs"]
        sense = norm["sense"]

        if sense is ConstraintSense.LESS_EQUAL:
            # ── slack variable: +1 in row ──
            slack_name = f"s{row + 1}"
            col = extra_col
            extra_col += 1

            A[row, col] = 1.0
            slack_indices.append(col)
            variable_names.append(slack_name)
            variable_info.append(VariableInfo(
                name=slack_name,
                role=VariableRole.SLACK,
                column_index=col,
            ))
            # Slack variable serves as initial basis column for this row
            phase1_basis_indices.append(col)

        elif sense is ConstraintSense.GREATER_EQUAL:
            # ── surplus variable: -1 in row ──
            surplus_name = f"e{row + 1}"
            e_col = extra_col
            extra_col += 1

            A[row, e_col] = -1.0
            surplus_indices.append(e_col)
            variable_names.append(surplus_name)
            variable_info.append(VariableInfo(
                name=surplus_name,
                role=VariableRole.SURPLUS,
                column_index=e_col,
            ))

            # ── artificial variable: +1 in row ──
            art_name = f"a{row + 1}"
            a_col = extra_col
            extra_col += 1

            A[row, a_col] = 1.0
            artificial_indices.append(a_col)
            phase1_c[a_col] = 1.0
            variable_names.append(art_name)
            variable_info.append(VariableInfo(
                name=art_name,
                role=VariableRole.ARTIFICIAL,
                column_index=a_col,
            ))

            # Artificial variable serves as initial basis column for this row
            phase1_basis_indices.append(a_col)
            requires_artificial.append(row)

        else:
            # ── equality constraint: artificial variable: +1 in row ──
            art_name = f"a{row + 1}"
            a_col = extra_col
            extra_col += 1

            A[row, a_col] = 1.0
            artificial_indices.append(a_col)
            phase1_c[a_col] = 1.0
            variable_names.append(art_name)
            variable_info.append(VariableInfo(
                name=art_name,
                role=VariableRole.ARTIFICIAL,
                column_index=a_col,
            ))

            # Artificial variable serves as initial basis column for this row
            phase1_basis_indices.append(a_col)
            requires_artificial.append(row)

    # ── upper-bound rows: y_i + s = U_i - L_i ────────────────────────
    # Each finite upper bound becomes an explicit <= row with a slack
    # variable, preserving the standard simplex pivot/ratio logic.
    for ub_row, (i, bound) in enumerate(upper_bound_rows):
        row = len(normalized_constrs) + ub_row
        A[row, i] = 1.0
        b[row] = bound

        slack_name = f"ub_s{row + 1}"
        col = extra_col
        extra_col += 1

        A[row, col] = 1.0
        slack_indices.append(col)
        variable_names.append(slack_name)
        variable_info.append(VariableInfo(
            name=slack_name,
            role=VariableRole.SLACK,
            column_index=col,
        ))
        phase1_basis_indices.append(col)

    # ── nonbasic indices = complement of basis ───────────────────────
    phase1_basis_set = set(phase1_basis_indices)
    phase1_nonbasic_indices = [
        j for j in range(n_total) if j not in phase1_basis_set
    ]

    # basis_indices equals phase1_basis_indices at initial construction
    basis_indices = list(phase1_basis_indices)
    nonbasic_indices = list(phase1_nonbasic_indices)

    model_sf = StandardFormModel(
        A=A,
        b=b,
        c=c,
        phase1_c=phase1_c,
        variable_names=variable_names,
        variable_info=variable_info,
        original_variable_mapping=original_variable_mapping,
        slack_indices=slack_indices,
        surplus_indices=surplus_indices,
        artificial_indices=artificial_indices,
        phase1_basis_indices=phase1_basis_indices,
        phase1_nonbasic_indices=phase1_nonbasic_indices,
        basis_indices=basis_indices,
        nonbasic_indices=nonbasic_indices,
        objective_sense=model.objective.sense,
        num_original_variables=n_orig,
        num_slack_variables=n_slack,
        num_surplus_variables=n_surplus,
        num_artificial_variables=n_artificial,
        requires_artificial=requires_artificial,
        upper_bounds=upper_bounds,
        variable_shifts=dict(variable_shifts),
        objective_constant=objective_constant,
    )

    # Validate the initial basis invariant
    model_sf.validate_basis()

    return model_sf
