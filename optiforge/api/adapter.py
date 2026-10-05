"""Adapter to convert API models to OptiForge internal model representation."""

from __future__ import annotations

from optiforge.api.models import (
    ConstraintInput,
    ObjectiveInput,
    SolveRequest,
    VariableInput,
)
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)


def convert_variable(api_var: VariableInput) -> Variable:
    """Convert API variable to OptiForge Variable."""
    var_type_map = {
        "continuous": VariableType.CONTINUOUS,
        "integer": VariableType.INTEGER,
        "binary": VariableType.BINARY,
    }

    upper_bound = api_var.upper_bound if api_var.upper_bound is not None else float("inf")

    return Variable(
        name=api_var.name,
        lower_bound=api_var.lower_bound,
        upper_bound=upper_bound,
        variable_type=var_type_map[api_var.variable_type],
    )


def convert_constraint(api_constr: ConstraintInput) -> Constraint:
    """Convert API constraint to OptiForge Constraint."""
    sense_map = {
        "<=": ConstraintSense.LESS_EQUAL,
        ">=": ConstraintSense.GREATER_EQUAL,
        "==": ConstraintSense.EQUAL,
    }

    return Constraint(
        name=api_constr.name,
        coefficients=dict(api_constr.coefficients),
        sense=sense_map[api_constr.sense],
        rhs=api_constr.rhs,
    )


def convert_objective(api_obj: ObjectiveInput) -> Objective:
    """Convert API objective to OptiForge Objective."""
    sense_map = {
        "minimize": ObjectiveSense.MINIMIZE,
        "maximize": ObjectiveSense.MAXIMIZE,
    }

    return Objective(
        coefficients=dict(api_obj.coefficients),
        sense=sense_map[api_obj.sense],
    )


def api_request_to_model(request: SolveRequest) -> Model:
    """Convert a SolveRequest to an OptiForge Model."""
    model = Model(name="api_model")

    # Add variables
    for api_var in request.variables:
        var = convert_variable(api_var)
        model.add_variable(var)

    # Add constraints
    for api_constr in request.constraints:
        constr = convert_constraint(api_constr)
        model.add_constraint(constr)

    # Set objective
    objective = convert_objective(request.objective)
    model.set_objective(objective)

    return model


def solution_to_response(solution, solve_time_ms: float | None = None, request: SolveRequest | None = None, model: Model | None = None) -> dict:
    """Convert OptiForge Solution to API response dict."""
    from optiforge.core.solution import SolutionStatus

    status_map = {
        SolutionStatus.OPTIMAL: "optimal",
        SolutionStatus.FEASIBLE: "feasible",
        SolutionStatus.INFEASIBLE: "infeasible",
        SolutionStatus.UNBOUNDED: "unbounded",
        SolutionStatus.ITERATION_LIMIT: "iteration_limit",
        SolutionStatus.UNKNOWN: "unknown",
    }

    status = status_map.get(solution.status, "unknown")
    
    # Status explanations
    status_explanations = {
        "optimal": "Optimal solution found and proven.",
        "feasible": "Feasible solution found, but optimality not proven.",
        "infeasible": "Problem is infeasible - no solution satisfies all constraints.",
        "unbounded": "Problem is unbounded - objective can be improved indefinitely.",
        "iteration_limit": "Solver reached iteration/node limit without proving optimality.",
        "unknown": "Solver status unknown.",
    }

    variable_values = []
    if request is not None and model is not None:
        # Include variable metadata from request and model (model has enforced bounds)
        var_type_map = {v.name: v.variable_type for v in request.variables}
        var_lb_map = {v.name: v.lower_bound for v in request.variables}
        # Use model's variable bounds (which have enforced bounds for binary variables)
        var_ub_map = {v.name: model.variables[v.name].upper_bound if v.name in model.variables else (v.upper_bound if v.upper_bound is not None else None) for v in request.variables}
    elif request is not None:
        # Fallback to request bounds if model not provided
        var_type_map = {v.name: v.variable_type for v in request.variables}
        var_lb_map = {v.name: v.lower_bound for v in request.variables}
        var_ub_map = {v.name: v.upper_bound if v.upper_bound is not None else None for v in request.variables}
    else:
        var_type_map = {}
        var_lb_map = {}
        var_ub_map = {}

    for name, value in solution.variable_values.items():
        var_type = var_type_map.get(name, "continuous")
        lower_bound = var_lb_map.get(name)
        upper_bound = var_ub_map.get(name)
        
        # Check integrality for integer/binary variables
        integrality_valid = None
        integrality_violation = None
        if var_type in ("integer", "binary"):
            nearest_int = round(value)
            violation = abs(value - nearest_int)
            integrality_violation = violation
            integrality_valid = violation <= 1e-6
            if var_type == "binary":
                integrality_valid = integrality_valid and (value == 0.0 or value == 1.0)
        
        variable_values.append({
            "name": name,
            "value": value,
            "variable_type": var_type,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "integrality_valid": integrality_valid,
            "integrality_violation": integrality_violation,
        })

    response = {
        "status": status,
        "objective_value": solution.objective_value,
        "variable_values": variable_values,
        "message": solution.message,
        "iterations": solution.iterations,
        "solve_time_ms": solve_time_ms,
        "status_explanation": status_explanations.get(status, "Unknown solver status."),
    }

    # Add MILP-specific fields if available
    if solution.node_count is not None:
        response["node_count"] = solution.node_count

    # Add numerical snapshots if available (from monitoring)
    if solution.numerical_snapshots is not None:
        response["numerical_snapshots"] = [
            {
                "iteration": snap.iteration,
                "node_id": snap.node_id,
                "primal_residual": snap.primal_residual,
                "dual_residual": snap.dual_residual,
                "constraint_violation": snap.constraint_violation,
                "bound_violation": snap.bound_violation,
                "integrality_violation": snap.integrality_violation,
                "pivot_magnitude": snap.pivot_magnitude,
                "condition_estimate": snap.condition_estimate,
                "warning_level": snap.warning_level.name
                if snap.warning_level is not None
                else None,
                "event_count": snap.event_count,
            }
            for snap in solution.numerical_snapshots
        ]

    # Add constraint evaluations if request is provided
    if request is not None:
        constraint_evaluations = []
        feasibility_tolerance = 1e-9
        for constr in request.constraints:
            # Compute LHS value
            lhs_value = sum(
                coeff * solution.variable_values.get(var_name, 0.0)
                for var_name, coeff in constr.coefficients.items()
            )
            
            # Compute slack/residual and violation based on constraint sense
            if constr.sense == "<=":
                slack = constr.rhs - lhs_value
                violation = max(0.0, lhs_value - constr.rhs)
                is_satisfied = lhs_value <= constr.rhs + feasibility_tolerance
                residual = None
            elif constr.sense == ">=":
                slack = lhs_value - constr.rhs
                violation = max(0.0, constr.rhs - lhs_value)
                is_satisfied = lhs_value >= constr.rhs - feasibility_tolerance
                residual = None
            else:  # "=="
                residual = abs(lhs_value - constr.rhs)
                violation = residual
                is_satisfied = residual <= feasibility_tolerance
                slack = None
            
            constraint_evaluations.append({
                "name": constr.name,
                "lhs_value": lhs_value,
                "sense": constr.sense,
                "rhs": constr.rhs,
                "slack": slack,
                "residual": residual,
                "violation": violation,
                "is_satisfied": is_satisfied,
            })
        
        response["constraint_evaluations"] = constraint_evaluations
        
        # Add feasibility info
        all_satisfied = all(ce["is_satisfied"] for ce in constraint_evaluations)
        max_violation = max((ce["violation"] for ce in constraint_evaluations), default=0.0)
        if all_satisfied:
            response["feasibility_info"] = f"All constraints satisfied (max violation: {max_violation:.2e})"
        else:
            violated = [ce["name"] for ce in constraint_evaluations if not ce["is_satisfied"]]
            response["feasibility_info"] = f"Constraints violated: {', '.join(violated)} (max violation: {max_violation:.2e})"

    return response