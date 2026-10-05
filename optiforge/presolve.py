"""Conservative LP presolve layer for OptiForge.

Implements Stage 5A presolve reductions that operate BEFORE standard-form
conversion and the Revised Simplex solver.  All reductions preserve feasibility,
preserve the original objective value (via an objective constant shift), and
record reversible metadata so postsolve can reconstruct the original solution.

The presolve loop iterates to a fixed point: bound tightening, singleton
processing, activity checks, fixed-variable elimination, and zero-row/redundancy
checks all run repeatedly until no changes occur, or INFEASIBLE/UNBOUNDED is
detected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from optiforge.model.constraint import ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import ObjectiveSense
from optiforge.model.variable import Variable

__all__ = [
    "PresolveResult",
    "presolve_model",
    "postsolve_solution",
]

# ---------------------------------------------------------------------------
# Numerical tolerance — shared across all reductions
# ---------------------------------------------------------------------------

FEASIBILITY_TOL: float = 1e-7


# ---------------------------------------------------------------------------
# PresolveResult — what presolve returns to the caller
# ---------------------------------------------------------------------------

@dataclass
class PresolveResult:
    """Result of presolve reduction.

    Parameters
    ----------
    status : SolutionStatus
        One of: optimal, infeasible, unbounded, unknown.
    reduced_model : Model
        A new Model instance representing the reduced problem.
        May be identical to the original if no reductions applied.
    variable_mapping : dict[str, str]
        Maps original variable names → reduced variable names (identity if no
        elimination occurred).  Used by postsolve to reconstruct original values.
    fixed_variables : dict[str, float]
        Variables that were fixed to a value during presolve.  Keys are original
        variable names; values are the fixed numbers.
    objective_constant : float
        Constant added to the reduced objective to recover the original
        objective value.  original_obj = reduced_obj + objective_constant.
    removed_rows : set[str]
        Constraint names that were eliminated as redundant.
    removed_variables : set[str]
        Variable names that were eliminated (fixed + substituted away).
    transformation_metadata : dict[str, Any]
        Extra information needed by postsolve (e.g. per-fixed-variable
        substitution records).
    statistics : dict[str, Any]
        Summary statistics about what presolve did.
    """

    status: str
    reduced_model: Model
    variable_mapping: dict[str, str] = field(default_factory=dict)
    fixed_variables: dict[str, float] = field(default_factory=dict)
    objective_constant: float = 0.0
    removed_rows: set[str] = field(default_factory=set)
    removed_variables: set[str] = field(default_factory=set)
    transformation_metadata: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: deep copy a Model (lightweight — only copies data, not methods)
# ---------------------------------------------------------------------------

def _copy_model(m: Model) -> Model:
    """Return a shallow-copy of *m* with the same variables/constraints/objective."""
    new_model = Model(name=m.name)
    # copy variables
    for vname, var in m.variables.items():
        new_model.add_variable(Variable(name=vname, lower_bound=var.lower_bound, upper_bound=var.upper_bound, variable_type=var.variable_type))
    # copy constraints
    for cname, con in m.constraints.items():
        new_model.add_constraint(
            Constraint(
                name=cname,
                coefficients=dict(con.coefficients),
                sense=con.sense,
                rhs=con.rhs,
            )
        )
    # copy objective
    if m.objective is not None:
        new_model.set_objective(
            Objective(
                coefficients=dict(m.objective.coefficients),
                sense=m.objective.sense,
            )
        )
    return new_model


# ---------------------------------------------------------------------------
# Stage 5A Presolve Implementations
# ---------------------------------------------------------------------------


def _check_contradictory_bounds(variable: Variable, tol: float = FEASIBILITY_TOL) -> str | None:
    """Return 'infeasible' if variable bounds are contradictory, else None."""
    if variable.lower_bound > variable.upper_bound + tol:
        return "infeasible"
    if abs(variable.lower_bound - variable.upper_bound) <= tol:
        return "fixed"
    return None


def _process_singleton_constraints(
    constraint: Constraint,
    var_names: list[str],
    var_map: dict[str, Variable],
    tol: float = FEASIBILITY_TOL,
) -> dict[str, Any] | None:
    """Process a singleton constraint.

    If the constraint has only one nonzero coefficient a_j, derive the
    corresponding bound or fixed value.

    Returns a dict with one of:
      - {"type": "fix", "var": var_name, "value": value}   if a_j * x_j = b
      - {"type": "bound", "var": var_name, "dir": "lower"|"upper", "value": value}
      - None   if nothing could be derived.
    """
    nz = [(var, coeff) for var, coeff in constraint.coefficients.items() if abs(coeff) > tol]
    if len(nz) != 1:
        return None

    (var_name, a), = nz
    var = var_map[var_name]
    rhs = constraint.rhs

    # a * x_j  ⊕  rhs  where ⊕ is the constraint sense
    if constraint.sense is ConstraintSense.LESS_EQUAL:
        # a * x_j <= rhs
        if abs(a) <= tol:
            return None
        if a > 0:
            # x_j <= rhs / a  -> tighten upper bound
            new_ub = rhs / a
            if new_ub < var.upper_bound:
                var.upper_bound = new_ub
                return {"type": "bound", "var": var_name, "dir": "upper", "value": new_ub}
            # If the derived bound is tighter than existing, we still record it
            # but do NOT remove the row.
            return {"type": "bound", "var": var_name, "dir": "upper", "value": new_ub}
        else:  # a < 0
            # dividing by negative reverses inequality:
            # a * x_j <= rhs  =>  x_j >= rhs / a
            new_lb = rhs / a  # a is negative, so rhs/a gives a lower bound
            if new_lb > var.lower_bound:
                var.lower_bound = new_lb
                return {"type": "bound", "var": var_name, "dir": "lower", "value": new_lb}
            return {"type": "bound", "var": var_name, "dir": "lower", "value": new_lb}

    elif constraint.sense is ConstraintSense.GREATER_EQUAL:
        # a * x_j >= rhs
        if abs(a) <= tol:
            return None
        if a > 0:
            # x_j >= rhs / a
            new_lb = rhs / a
            if new_lb > var.lower_bound:
                var.lower_bound = new_lb
                return {"type": "bound", "var": var_name, "dir": "lower", "value": new_lb}
            return {"type": "bound", "var": var_name, "dir": "lower", "value": new_lb}
        else:  # a < 0
            # a * x_j >= rhs  =>  x_j <= rhs / a  (reverse inequality)
            new_ub = rhs / a
            if new_ub < var.upper_bound:
                var.upper_bound = new_ub
                return {"type": "bound", "var": var_name, "dir": "upper", "value": new_ub}
            return {"type": "bound", "var": var_name, "dir": "upper", "value": new_ub}

    else:  # EQUAL
        # a * x_j = rhs
        if abs(a) <= tol:
            return None
        val = rhs / a
        # Fix the variable to this value
        var.lower_bound = val
        var.upper_bound = val
        return {"type": "fix", "var": var_name, "value": val}


def _process_zero_rows(
    constraint: Constraint,
    var_map: dict[str, Variable],
    tol: float = FEASIBILITY_TOL,
) -> str | None:
    """Check if a constraint is all-zero.

    Returns:
      - "redundant"  if the row can be removed.
      - "infeasible" if the row declares infeasibility.
      - None         if the row is not all-zero.
    """
    nz_coeffs = {var: coeff for var, coeff in constraint.coefficients.items() if abs(coeff) > tol}
    if not nz_coeffs:
        # All coefficients are zero
        sense = constraint.sense
        rhs = constraint.rhs
        if sense is ConstraintSense.LESS_EQUAL:
            # 0 <= rhs  -> always feasible (redundant)
            return "redundant"
        elif sense is ConstraintSense.GREATER_EQUAL:
            # 0 >= rhs  -> feasible only if rhs <= 0
            if rhs <= tol:
                return "redundant"
            else:
                return "infeasible"
        elif sense is ConstraintSense.EQUAL:
            # 0 = rhs  -> feasible only if rhs == 0
            if abs(rhs) <= tol:
                return "redundant"
            else:
                return "infeasible"
    return None


def _process_zero_columns(
    model: Model,
    var_names: list[str],
    var_map: dict[str, Variable],
    tol: float = FEASIBILITY_TOL,
) -> dict[str, Any] | None:
    """Process variables that appear in NO constraint (zero column).

    For MIN: c_j > 0 -> choose lower bound, c_j < 0 -> choose upper bound,
             c_j = 0 -> any feasible value.
    For MAX: reverse direction.

    Returns a dict describing the action, or None if the variable appears in
    at least one constraint.
    """
    # Check if this variable appears in any constraint
    appears_in_constraint = False
    for con in model.constraints.values():
        for vname in con.coefficients:
            if vname in var_map:
                appears_in_constraint = True
                break
        if appears_in_constraint:
            break

    if appears_in_constraint:
        return None

    # Variable is a zero column — only objective + bounds determine it
    # Find the variable's objective coefficient
    var_name = var_names[0]  # will be set per variable
    # This function is called per variable; the caller must track which variable.
    # For now, return None to signal "not a zero column in this pass"
    # (the actual per-variable logic is handled in the iteration loop).
    return None


def _activity_bounds_check(
    model: Model,
    tol: float = FEASIBILITY_TOL,
) -> dict[str, Any] | None:
    """Check activity bounds for all constraints.

    For a constraint a^T x <= b:
      L = sum min(a_i * l_i, a_i * u_i)
      U = sum max(a_i * l_i, a_i * u_i)
      If L > b + tol       -> INFEASIBLE
      If U <= b + tol      -> REDUNDANT

    For a constraint a^T x >= b:
      If U < b - tol       -> INFEASIBLE
      If L >= b - tol      -> REDUNDANT

    For a constraint a^T x = b:
      If L > b + tol OR U < b - tol -> INFEASIBLE
      If |L - b| <= tol AND |U - b| <= tol -> REDUNDANT

    Returns dict with 'action' key: "infeasible", "redundant", or None.
    Also returns the tightened/reduced bounds in metadata.
    """
    for con_name, constraint in model.constraints.items():
        a = np.array([coeff for _, coeff in constraint.coefficients.items()], dtype=np.float64)
        # We need l_i and u_i for each variable in order.
        # Build ordered lists matching the coefficients dict order.
        var_names_in_con = list(constraint.coefficients.keys())
        lhs_coeffs = []
        l_vals = []
        u_vals = []
        for vn in var_names_in_con:
            var = model.variables[vn]
            l_vals.append(var.lower_bound)
            u_vals.append(var.upper_bound)
            lhs_coeffs.append(constraint.coefficients[vn])

        rhs = float(constraint.rhs)
        sense = constraint.sense

        # Compute L = sum min(a_i * l_i, a_i * u_i)
        # and U = sum max(a_i * l_i, a_i * u_i)
        # Handle infinite bounds safely (avoid inf*0)
        L = 0.0
        U = 0.0
        for i, coeff in enumerate(lhs_coeffs):
            li = float(l_vals[i])
            ui = float(u_vals[i])
            ai = float(coeff)

            # min(a*l, a*u) and max(a*l, a*u) with safe inf handling
            if ai == 0.0:
                contrib = 0.0
            elif ai > 0:
                # a*l and a*u: min is a*l if l is finite, etc.
                al = ai * li if not (math.isinf(ai) and math.isinf(li) and li < 0) else _safe_inf_prod(ai, li)
                au = ai * ui if not (math.isinf(ai) and math.isinf(ui) and ui > 0) else _safe_inf_prod(ai, ui)
                # Actually simpler: just compute with care
                al = ai * li
                au = ai * ui
                # If ai > 0 and both bounds are finite, min = a*l, max = a*u
                # If bounds are infinite, we handle below
                if math.isfinite(li) and math.isfinite(ui):
                    contrib_min = min(al, au)
                    contrib_max = max(al, au)
                else:
                    # With infinities, we need careful handling
                    # min(a*l, a*u) when a>0: if l is -inf, min is -inf; if u is +inf, max is +inf
                    # We'll use a conservative approach
                    contrib_min = _safe_min_prod(ai, li, ui)
                    contrib_max = _safe_max_prod(ai, li, ui)
            else:  # ai < 0
                al = ai * li
                au = ai * ui
                if math.isfinite(li) and math.isfinite(ui):
                    contrib_min = min(al, au)
                    contrib_max = max(al, au)
                else:
                    contrib_min = _safe_min_prod(ai, li, ui)
                    contrib_max = _safe_max_prod(ai, li, ui)

            L += contrib_min
            U += contrib_max

        # Now check based on sense
        action = None
        if sense is ConstraintSense.LESS_EQUAL:
            if L > rhs + tol:
                action = "infeasible"
            elif U <= rhs + tol:
                action = "redundant"
        elif sense is ConstraintSense.GREATER_EQUAL:
            if U < rhs - tol:
                action = "infeasible"
            elif L >= rhs - tol:
                action = "redundant"
        elif sense is ConstraintSense.EQUAL:
            if L > rhs + tol or U < rhs - tol:
                action = "infeasible"
            elif abs(L - rhs) <= tol and abs(U - rhs) <= tol:
                action = "redundant"

        if action is not None:
            return {"action": action, "constraint_name": con_name, "L": float(L), "U": float(U)}

    return None


def _safe_min_prod(ai: float, li: float, ui: float) -> float:
    """Compute min(ai*li, ai*ui) safely with infinite bounds."""
    # When ai > 0: min is ai*li (since li <= ui), max is ai*ui
    # When ai < 0: min is ai*ui (since ai*ui <= ai*li because ui >= li and ai < 0)
    if ai > 0:
        # min = ai * li, but handle li = -inf
        if math.isinf(li) and li < 0:
            return float("-inf")
        if math.isinf(ui) and ui > 0:
            # ui is +inf, ai*ui is +inf, so min is ai*li
            return ai * li if math.isfinite(li) else float("-inf")
        return ai * li
    else:  # ai < 0
        # min = ai * ui (reversed because ai < 0)
        if math.isinf(ui) and ui > 0:
            return float("-inf")  # ai * +inf = -inf when ai < 0
        if math.isinf(li) and li < 0:
            # ai * -inf = +inf when ai < 0, but li is the lower bound...
            # Actually if li = -inf and ai < 0, then ai*li = +inf
            # and ai*ui could be anything. The min would be the other term.
            # Conservative: return ai*ui
            pass
        return ai * ui


def _safe_max_prod(ai: float, li: float, ui: float) -> float:
    """Compute max(ai*li, ai*ui) safely with infinite bounds."""
    if ai > 0:
        # max = ai * ui
        if math.isinf(ui) and ui > 0:
            return float("inf")
        if math.isinf(li) and li < 0:
            # ai * -inf = -inf when ai > 0, but ui could be larger
            return ai * ui
        return ai * ui
    else:  # ai < 0
        # max = ai * li (reversed)
        if math.isinf(li) and li < 0:
            return float("inf")  # ai * -inf = +inf when ai < 0
        return ai * li


def _bound_validation(
    model: Model,
    tol: float = FEASIBILITY_TOL,
) -> tuple[dict[str, float] | None, str | None]:
    """Check all variable bounds for contradictions and fixed variables.

    Returns
    -------
    fixed : dict[str, float] | None
        Maps variable name -> fixed value if fixed, or None if no variables fixed.
    status : str | None
        "infeasible" if any variable has contradictory bounds, else None.
    """
    fixed: dict[str, float] = {}
    for var_name, var in model.variables.items():
        result = _check_contradictory_bounds(var, tol)
        if result == "infeasible":
            return None, "infeasible"
        if result == "fixed":
            fixed[var_name] = var.lower_bound  # lb == ub within tol
    return fixed, None


def _fixed_variable_elimination(
    model: Model,
    fixed: dict[str, float],
    tol: float = FEASIBILITY_TOL,
) -> tuple[Model, dict[str, float], set[str], set[str], dict[str, Any]]:
    """Eliminate fixed variables from the model.

    For each fixed variable x_j = v:
      - substitute a_j * v into every constraint RHS
      - remove x_j from reduced model
      - add c_j * v to objective constant
      - record reversible metadata so postsolve restores x_j = v

    Returns
    -------
    new_model : Model
        Model with fixed variables eliminated.
    objective_constant : float
        Sum of c_j * v over all fixed variables.
    fixed_variables : set[str]
        Names of fixed variables (for metadata).
    removed_variables : set[str]
        Names of variables removed from the model.
    metadata : dict[str, Any]
        Per-fixed-variable substitution records.
    """
    new_model = _copy_model(model)
    objective_constant = 0.0
    fixed_variables: set[str] = set()
    removed_variables: set[str] = set()
    metadata: dict[str, Any] = {}

    # We need to process variables in a deterministic order
    var_names = sorted(new_model.variables.keys())

    # Build a mapping of variable -> coefficient in each constraint
    # and in the objective
    for var_name in var_names:
        if var_name not in fixed:
            continue

        v = fixed[var_name]
        var_obj = model.variables[var_name]
        cj = model.objective.coefficients.get(var_name, 0.0)

        # Add c_j * v to objective constant
        objective_constant += cj * v

        fixed_variables.add(var_name)
        removed_variables.add(var_name)

        # Remove variable from objective coefficients
        # (we'll rebuild the objective later)
        # Substitute into constraints: for each constraint, RHS -= a_j * v
        constraints_to_remove = []
        for con_name, constraint in new_model.constraints.items():
            if var_name in constraint.coefficients:
                aj = constraint.coefficients[var_name]
                new_model.constraints[con_name].rhs = float(constraint.rhs) - aj * v
                # Remove the variable from coefficients
                del constraint.coefficients[var_name]

        # Remove variable from objective
        if var_name in new_model.objective.coefficients:
            del new_model.objective.coefficients[var_name]

        # Record metadata for postsolve
        metadata[var_name] = {"fixed_value": v, "original_rhs_adjustment": aj * v if var_name in model.constraints else 0.0}

        # Remove the variable from the model
        if var_name in new_model._variables:
            del new_model._variables[var_name]

    # Rebuild the model with remaining variables
    # Create a new model with the remaining variables
    remaining_vars = {name: var for name, var in new_model.variables.items() if name not in removed_variables}
    remaining_model = Model(name=new_model.name)
    for vname, var in remaining_vars.items():
        remaining_model.add_variable(Variable(name=vname, lower_bound=var.lower_bound, upper_bound=var.upper_bound, variable_type=var.variable_type))

    # Copy constraints with adjusted RHS
    for con_name, constraint in new_model.constraints.items():
        # Filter coefficients to only include remaining variables
        new_coeffs = {k: v for k, v in constraint.coefficients.items() if k in remaining_vars}
        if new_coeffs:  # only add if there are remaining coefficients
            remaining_model.add_constraint(
                Constraint(name=con_name, coefficients=new_coeffs, sense=constraint.sense, rhs=constraint.rhs)
            )
        # If no remaining coefficients, the constraint is redundant (all vars fixed/removed)
        # and we just skip it

    # Copy objective with remaining coefficients
    new_obj_coeffs = {k: v for k, v in new_model.objective.coefficients.items() if k in remaining_vars}
    if new_obj_coeffs:
        remaining_model.set_objective(
            Objective(coefficients=new_obj_coeffs, sense=new_model.objective.sense)
        )

    # Add the objective constant to... actually we track it separately
    # The reduced model's objective doesn't include the constant; postsolve adds it back

    return remaining_model, objective_constant, fixed_variables, removed_variables, metadata


def _remove_redundant_rows(
    model: Model,
) -> tuple[Model, set[str]]:
    """Remove constraints that are redundant (all-zero coefficients or activity bounds).

    Returns the new model and the set of removed constraint names.
    """
    new_model = _copy_model(model)
    removed_rows: set[str] = []

    # First pass: check all-zero rows
    rows_to_remove: set[str] = set()
    for con_name, constraint in new_model.constraints.items():
        result = _process_zero_rows(constraint, new_model.variables)
        if result is not None:
            rows_to_remove.add(con_name)

    # Second pass: activity bounds check
    activity_result = _activity_bounds_check(new_model)
    if activity_result is not None:
        if activity_result["action"] == "infeasible":
            return new_model, {"infeasible": True}  # signal infeasibility
        elif activity_result["action"] == "redundant":
            removed_rows.add(activity_result["constraint_name"])

    # Remove the marked rows
    for con_name in rows_to_remove:
        if con_name in new_model._constraints:
            del new_model._constraints[con_name]
            removed_rows.add(con_name)

    return new_model, removed_rows


# ---------------------------------------------------------------------------
# Main presolve function
# ---------------------------------------------------------------------------


def presolve_model(
    model: Model,
    max_passes: int = 20,
    tol: float = FEASIBILITY_TOL,
) -> PresolveResult:
    """Run the full presolve reduction pipeline on *model*.

    The pipeline iterates to a fixed point, applying reductions in order:
      1. Bound validation (contradictory/fixed bounds)
      2. Fixed variable elimination
      3. Zero row / redundant row checks
      4. Singleton constraint processing
      5. Zero column handling
      6. Activity bounds checks
      7. Bound propagation

    Reductions are repeated until no changes occur in a full pass, or
    INFEASIBLE/UNBOUNDED is detected.

    Parameters
    ----------
    model : Model
        The original LP model.
    max_passes : int
        Maximum number of fixed-point iterations. Default 20.
    tol : float
        Feasibility tolerance for bound checks. Default 1e-7.

    Returns
    -------
    PresolveResult
        Encapsulates the reduced model, status, variable mapping,
        fixed variables, objective constant, removed rows, and statistics.
    """
    # Working copies
    working_model = _copy_model(model)
    variable_mapping: dict[str, str] = {vname: vname for vname in working_model.variables}
    fixed_variables: dict[str, float] = {}
    objective_constant = 0.0
    removed_rows: set[str] = set()
    removed_variables: set[str] = set()
    transformation_metadata: dict[str, Any] = {}
    all_fixed: dict[str, float] = {}

    # Statistics
    stats = {
        "original_vars": working_model.num_variables,
        "original_constraints": working_model.num_constraints,
        "fixed_vars": 0,
        "removed_rows": 0,
        "tightened_bounds": 0,
        "passes": 0,
        "detected_infeasible": False,
        "detected_unbounded": False,
    }

    for pass_num in range(1, max_passes + 1):
        stats["passes"] = pass_num
        changes_occurred = False

        # --- 1. Bound validation ---
        fixed, status = _bound_validation(working_model, tol)
        if status == "infeasible":
            stats["detected_infeasible"] = True
            result_status = "infeasible"
            # Build a minimal result for infeasible case
            infeasible_model = _copy_model(working_model)
            # Remove all variables to create an empty model
            infeasible_model = Model(name=working_model.name)
            # We need at least one variable for the model to be valid, but for infeasible
            # we return the model as-is and mark status
            return PresolveResult(
                status=result_status,
                reduced_model=infeasible_model,
                variable_mapping={},
                fixed_variables={},
                objective_constant=0.0,
                removed_rows=set(),
                removed_variables=set(),
                transformation_metadata={},
                statistics=stats,
            )

        if fixed:
            changes_occurred = True
            # Eliminate fixed variables
            working_model, obj_const, fixed_vars, removed_vars, meta = _fixed_variable_elimination(
                working_model, fixed, tol
            )
            objective_constant += obj_const
            fixed_variables.update(fixed_vars)
            removed_variables.update(removed_vars)
            transformation_metadata.update(meta)
            # Update variable mapping: eliminated vars map to "eliminated"
            for v in removed_vars:
                if v in variable_mapping:
                    variable_mapping[v] = "__eliminated__"
            stats["fixed_vars"] = len(fixed_variables)
            stats["removed_variables"] = len(removed_variables)
            changes_occurred = True

        # --- 2. Zero rows (all-zero coefficients) ---
        # We check after fixed variable elimination because RHS may have changed
        model_before_zero = _copy_model(working_model)
        working_model, zero_removed = _remove_redundant_rows(working_model)
        if zero_removed:
            if "infeasible" not in str(zero_removed):
                removed_rows.update(zero_removed)
            changes_occurred = True
            stats["removed_rows"] = len(removed_rows)

        # --- 3. Singleton constraint processing ---
        if working_model.num_constraints > 0:
            for con_name in list(working_model.constraints.keys()):
                constraint = working_model.constraints[con_name]
                result = _process_singleton_constraints(
                    constraint,
                    list(constraint.coefficients.keys()),
                    working_model.variables,
                    tol,
                )
                if result is not None:
                    changes_occurred = True
                    if result["type"] == "fix":
                        var_name = result["var"]
                        val = result["value"]
                        # Fix the variable bound
                        if var_name in working_model.variables:
                            var = working_model.variables[var_name]
                            var.lower_bound = val
                            var.upper_bound = val
                            all_fixed[var_name] = val
                            fixed_variables[var_name] = val
                            removed_variables.add(var_name)
                            # Update mapping
                            if var_name in variable_mapping:
                                variable_mapping[var_name] = "__fixed__"
                            # Remove from constraints and objective
                            # Remove from all constraints
                            for cn in list(working_model.constraints.keys()):
                                if var_name in working_model.constraints[cn].coefficients:
                                    del working_model.constraints[cn].coefficients[var_name]
                            # Remove from objective
                            if var_name in working_model.objective.coefficients:
                                del working_model.objective.coefficients[var_name]
                            # Update objective constant
                            cj = model.objective.coefficients.get(var_name, 0.0)
                            objective_constant += cj * val
                            stats["fixed_vars"] = len(fixed_variables)
                            changes_occurred = True
                    elif result["type"] == "bound":
                        var_name = result["var"]
                        dir_ = result["dir"]
                        val = result["value"]
                        if var_name in working_model.variables:
                            var = working_model.variables[var_name]
                            if dir_ == "upper":
                                if val < var.upper_bound:
                                    var.upper_bound = val
                                    stats["tightened_bounds"] += 1
                            elif dir_ == "lower":
                                if val > var.lower_bound:
                                    var.lower_bound = val
                                    stats["tightened_bounds"] += 1
                            changes_occurred = True

        # --- 4. Activity bounds check ---
        activity_result = _activity_bounds_check(working_model, tol)
        if activity_result is not None:
            changes_occurred = True
            if activity_result["action"] == "infeasible":
                stats["detected_infeasible"] = True
                result_status = "infeasible"
                infeasible_model = Model(name=working_model.name)
                return PresolveResult(
                    status=result_status,
                    reduced_model=infeasible_model,
                    variable_mapping={},
                    fixed_variables={},
                    objective_constant=objective_constant,
                    removed_rows=removed_rows,
                    removed_variables=removed_variables,
                    transformation_metadata=transformation_metadata,
                    statistics=stats,
                )
            elif activity_result["action"] == "redundant":
                removed_rows.add(activity_result["constraint_name"])
                # Also remove from working model
                con_name = activity_result["constraint_name"]
                if con_name in working_model._constraints:
                    del working_model._constraints[con_name]

        # --- 5. Zero column handling (variable appears in no constraint) ---
        # Check each variable
        if working_model.num_variables > 0:
            for var_name in list(working_model.variables.keys()):
                var = working_model.variables[var_name]
                # Check if this variable appears in any constraint
                appears_in_con = False
                for con in working_model.constraints.values():
                    if var_name in con.coefficients:
                        appears_in_con = True
                        break

                if not appears_in_con:
                    # Zero column variable — only objective + bounds determine it
                    cj = working_model.objective.coefficients.get(var_name, 0.0)
                    sense = working_model.objective.sense

                    # For MIN: cj > 0 -> lower bound, cj < 0 -> upper bound, cj = 0 -> any feasible
                    # For MAX: reverse
                    if sense is ObjectiveSense.MINIMIZE:
                        if cj > 0:
                            # Choose lower bound
                            if var.lower_bound < var.upper_bound:
                                # Fix variable to lower bound
                                var.upper_bound = var.lower_bound
                                all_fixed[var_name] = var.lower_bound
                                fixed_variables[var_name] = var.lower_bound
                                removed_variables.add(var_name)
                                if var_name in variable_mapping:
                                    variable_mapping[var_name] = "__fixed__"
                                # Remove from objective
                                del working_model.objective.coefficients[var_name]
                                objective_constant += cj * var.lower_bound
                                stats["fixed_vars"] = len(fixed_variables)
                                changes_occurred = True
                        elif cj < 0:
                            # Choose upper bound
                            if var.lower_bound < var.upper_bound:
                                var.lower_bound = var.upper_bound
                                all_fixed[var_name] = var.upper_bound
                                fixed_variables[var_name] = var.upper_bound
                                removed_variables.add(var_name)
                                if var_name in variable_mapping:
                                    variable_mapping[var_name] = "__fixed__"
                                # Remove from objective
                                del working_model.objective.coefficients[var_name]
                                objective_constant += cj * var.upper_bound
                                stats["fixed_vars"] = len(fixed_variables)
                                changes_occurred = True
                        # cj == 0: any feasible value, do nothing (leave as is)
                    else:  # MAXIMIZE
                        if cj > 0:
                            # Choose upper bound (reverse for MAX)
                            if var.lower_bound < var.upper_bound:
                                var.lower_bound = var.upper_bound
                                all_fixed[var_name] = var.upper_bound
                                fixed_variables[var_name] = var.upper_bound
                                removed_variables.add(var_name)
                                if var_name in variable_mapping:
                                    variable_mapping[var_name] = "__fixed__"
                                # Remove from objective
                                del working_model.objective.coefficients[var_name]
                                objective_constant += cj * var.upper_bound
                                stats["fixed_vars"] = len(fixed_variables)
                                changes_occurred = True
                        elif cj < 0:
                            # Choose lower bound (reverse for MAX)
                            if var.lower_bound < var.upper_bound:
                                var.upper_bound = var.lower_bound
                                all_fixed[var_name] = var.lower_bound
                                fixed_variables[var_name] = var.lower_bound
                                removed_variables.add(var_name)
                                if var_name in variable_mapping:
                                    variable_mapping[var_name] = "__fixed__"
                                # Remove from objective
                                del working_model.objective.coefficients[var_name]
                                objective_constant += cj * var.lower_bound
                                stats["fixed_vars"] = len(fixed_variables)
                                changes_occurred = True

        # --- 6. Bound propagation ---
        # For each constraint a^T x <= b (or >= or =), tighten variable bounds
        # using the activity bound logic
        if working_model.num_constraints > 0 and working_model.num_variables > 0:
            prop_changes = _propagate_bounds(working_model, tol)
            if prop_changes:
                changes_occurred = True
                stats["tightened_bounds"] += prop_changes

        # If no changes occurred in this pass, we've reached fixed point
        if not changes_occurred:
            # Final check: are there still any fixed variables we missed?
            # (should have been handled in earlier passes)
            break

    # Build the final result
    stats["fixed_vars"] = len(fixed_variables)
    stats["removed_rows"] = len(removed_rows)

    # Determine final status
    if stats["detected_infeasible"]:
        result_status = "infeasible"
    elif stats["passes"] >= max_passes:
        result_status = "unknown"  # didn't converge within max passes
    else:
        result_status = "optimal"  # no reductions detected issues

    # Build reduced model - need to ensure it's valid
    if working_model.num_variables == 0 and working_model.num_constraints == 0:
        # Empty model case
        reduced_model = Model(name=working_model.name)
    else:
        reduced_model = working_model

    return PresolveResult(
        status=result_status,
        reduced_model=reduced_model,
        variable_mapping=variable_mapping,
        fixed_variables=fixed_variables,
        objective_constant=objective_constant,
        removed_rows=removed_rows,
        removed_variables=removed_variables,
        transformation_metadata=transformation_metadata,
        statistics=stats,
    )


def _propagate_bounds(model: Model, tol: float = FEASIBILITY_TOL) -> int:
    """Perform bound propagation across all constraints.

    For each constraint a^T x <= b (or >= or =), tighten variable bounds using
    the minimum/maximum possible contribution of other variables.

    Returns the number of bound tightenings performed.
    """
    tightenings = 0

    # Collect all variables and their current bounds
    var_names = list(model.variables.keys())

    for con_name, constraint in model.constraints.items():
        # Get coefficients in order of var_names
        coeffs_list = []
        var_order = []
        for vn in var_names:
            if vn in constraint.coefficients:
                coeffs_list.append(constraint.coefficients[vn])
                var_order.append(vn)
            else:
                coeffs_list.append(0.0)
                var_order.append(vn)

        rhs = float(constraint.rhs)
        sense = constraint.sense

        # For each variable j in this constraint, tighten its bounds
        for j in range(len(var_order)):
            vn = var_order[j]
            aj = float(coeffs_list[j])
            if abs(aj) <= tol:
                # Skip zero coefficients
                continue

            var = model.variables[vn]
            lj = float(var.lower_bound)
            uj = float(var.upper_bound)

            # Compute min_other = sum(i!=j) min(a_i * l_i, a_i * u_i)
            # and max_other = sum(i!=j) max(a_i * l_i, a_i * u_i)
            min_other = 0.0
            max_other = 0.0

            for i in range(len(var_order)):
                if i == j:
                    continue
                ai = float(coeffs_list[i])
                if abs(ai) <= tol:
                    continue
                li = float(model.variables[var_order[i]].lower_bound)
                ui = float(model.variables[var_order[i]].upper_bound)

                # min(a*l, a*u)
                if ai > 0:
                    # min = a*l (if l is finite), otherwise handle inf
                    if math.isfinite(li) and math.isfinite(ui):
                        contrib_min = min(ai * li, ai * ui)
                        contrib_max = max(ai * li, ai * ui)
                    else:
                        # With infinities, use safe functions
                        contrib_min = _safe_min_prod(ai, li, ui)
                        contrib_max = _safe_max_prod(ai, li, ui)
                else:  # ai < 0
                    # min = a*u (reversed), max = a*l (reversed)
                    if math.isfinite(li) and math.isfinite(ui):
                        contrib_min = min(ai * li, ai * ui)
                        contrib_max = max(ai * li, ai * ui)
                    else:
                        contrib_min = _safe_min_prod(ai, li, ui)
                        contrib_max = _safe_max_prod(ai, li, ui)

                min_other += contrib_min
                max_other += contrib_max

            # Now apply the tightening rules
            if sense is ConstraintSense.LESS_EQUAL:
                # a_j * x_j <= rhs - min_other
                # If a_j > 0: x_j <= (rhs - min_other) / a_j
                # If a_j < 0: x_j >= (rhs - min_other) / a_j  (reverse)
                remaining_rhs = rhs - min_other

                if aj > 0:
                    new_ub = remaining_rhs / aj
                    if new_ub < var.upper_bound:
                        var.upper_bound = new_ub
                        tightenings += 1
                else:  # aj < 0
                    new_lb = remaining_rhs / aj  # dividing by negative
                    if new_lb > var.lower_bound:
                        var.lower_bound = new_lb
                        tightenings += 1

            elif sense is ConstraintSense.GREATER_EQUAL:
                # a_j * x_j >= rhs - max_other
                # If a_j > 0: x_j >= (rhs - max_other) / a_j
                # If a_j < 0: x_j <= (rhs - max_other) / a_j  (reverse)
                remaining_rhs = rhs - max_other

                if aj > 0:
                    new_lb = remaining_rhs / aj
                    if new_lb > var.lower_bound:
                        var.lower_bound = new_lb
                        tightenings += 1
                else:  # aj < 0
                    new_ub = remaining_rhs / aj  # dividing by negative reverses
                    if new_ub < var.upper_bound:
                        var.upper_bound = new_ub
                        tightenings += 1

            elif sense is ConstraintSense.EQUAL:
                # For equality constraints, we can derive bounds from both sides
                # a_j * x_j = rhs - contribution_of_others
                # This is more complex; for now, we derive a tight bound from the
                # min/max possible contribution
                # We'll treat it as: the variable's value is constrained such that
                # the total can still equal rhs
                # Simple approach: compute what x_j must be if others at their extremes
                remaining_rhs_minus = rhs - max_other  # when others contribute max
                remaining_rhs_plus = rhs - min_other  # when others contribute min

                if aj > 0:
                    # x_j = (rhs - others) / aj
                    # Others can contribute between min_other and max_other
                    # So x_j is between (rhs - max_other)/aj and (rhs - min_other)/aj
                    new_lb = remaining_rhs_plus / aj  # when others contribute min -> x_j max
                    new_ub = remaining_rhs_minus / aj  # when others contribute max -> x_j min
                    # Actually let's think again...
                    # If others contribute min (smallest), then a_j*x_j = rhs - min_other
                    # so x_j = (rhs - min_other)/a_j (this is the MAXIMUM x_j can be)
                    # If others contribute max (largest), then a_j*x_j = rhs - max_other
                    # so x_j = (rhs - max_other)/a_j (this is the MINIMUM x_j can be)
                    # Since a_j > 0, min_other <= max_other, so (rhs - min_other)/a_j >= (rhs - max_other)/a_j
                    # So x_j is in [(rhs - max_other)/a_j, (rhs - min_other)/a_j]
                    computed_lb = remaining_rhs_minus / aj  # this is actually the max
                    computed_ub = remaining_rhs_plus / aj  # this is actually the min
                    # Wait, I'm confusing myself. Let me be more careful.
                    # a_j * x_j + sum_{i!=j} a_i * x_i = rhs
                    # a_j * x_j = rhs - sum_{i!=j} a_i * x_i
                    # The sum_{i!=j} a_i * x_i ranges between min_other and max_other
                    # So a_j * x_j ranges between rhs - max_other and rhs - min_other
                    # Since a_j > 0, x_j ranges between (rhs - max_other)/a_j and (rhs - min_other)/a_j
                    # Lower bound on x_j: (rhs - max_other)/a_j
                    # Upper bound on x_j: (rhs - min_other)/a_j
                    computed_lb = (rhs - max_other) / aj
                    computed_ub = (rhs - min_other) / aj
                else:  # aj < 0
                    # a_j * x_j = rhs - sum_{i!=j} a_i * x_i
                    # a_j is negative, so when we divide, inequalities reverse
                    # sum_{i!=j} a_i * x_i ranges between min_other and max_other
                    # a_j * x_j ranges between rhs - max_other and rhs - min_other
                    # Since a_j < 0, dividing reverses: x_j ranges between (rhs - min_other)/a_j and (rhs - max_other)/a_j
                    # But since a_j < 0, (rhs - min_other)/a_j <= (rhs - max_other)/a_j
                    # Lower bound: (rhs - min_other)/a_j
                    # Upper bound: (rhs - max_other)/a_j
                    computed_lb = (rhs - min_other) / aj
                    computed_ub = (rhs - max_other) / aj

                # Now tighten the variable bounds
                if computed_lb > var.lower_bound:
                    var.lower_bound = computed_lb
                    tightenings += 1
                if computed_ub < var.upper_bound:
                    var.upper_bound = computed_ub
                    tightenings += 1

    return tightenings


# ---------------------------------------------------------------------------
# Postsolve
# ---------------------------------------------------------------------------


def postsolve_solution(
    reduced_variable_values: dict[str, float],
    presolve_result: PresolveResult,
) -> dict[str, float]:
    """Reconstruct the original variable values from a reduced solution.

    Parameters
    ----------
    reduced_variable_values : dict[str, float]
        Variable values from the solver on the reduced model.
    presolve_result : PresolveResult
        The presolve result containing mapping and fixed variable information.

    Returns
    -------
    dict[str, float]
        Mapping of original variable names → values in the original space.
    """
    restored: dict[str, float] = {}

    # Start with the reduced variable values, mapping names
    for var_name, val in reduced_variable_values.items():
        # Check if this variable was eliminated
        if var_name in presolve_result.fixed_variables:
            # It was fixed; the reduced value should be 0 or the fixed value
            # Use the fixed value from metadata
            if var_name in presolve_result.transformation_metadata:
                meta = presolve_result.transformation_metadata[var_name]
                restored[var_name] = meta.get("fixed_value", val)
            else:
                restored[var_name] = presolve_result.fixed_variables.get(var_name, val)
        elif var_name in presolve_result.variable_mapping:
            # Map to the reduced variable name
            reduced_name = presolve_result.variable_mapping[var_name]
            if reduced_name in reduced_variable_values:
                restored[var_name] = reduced_variable_values[reduced_name]
            else:
                restored[var_name] = val
        else:
            # Not eliminated, name is the same
            restored[var_name] = val

    # Now apply fixed variable values (variables that were fixed during presolve)
    for var_name, fixed_val in presolve_result.fixed_variables.items():
        restored[var_name] = fixed_val

    # Apply objective constant tracking is handled by the caller (Model.solve)
    # The original objective = reduced objective + objective_constant

    return restored