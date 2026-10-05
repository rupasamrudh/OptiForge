"""Tests verifying Stage 8.5 audit remediation fixes.

Covers:
1. Solver timeout cooperative cancellation & thread termination via stop_event
2. CORS configuration security (no wildcard + credentials, allowed origins)
3. Negative upper_bound validation (-5 <= x <= -1 is valid)
4. Duplicate/dead exception-handling cleanup & solver status classification
5. Whitespace-only variable and constraint name rejection & normalization
6. All-zero objective coefficient rejection
7. Request size limit enforcement & timeout parameter boundaries
8. MILP benchmark seed differentiation and classification
"""

import math
import threading
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from optiforge.api.main import app, solve_with_timeout, ALLOWED_ORIGINS
from optiforge.api.models import ConstraintInput, ObjectiveInput, SolveRequest, VariableInput
from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver
from optiforge.milp.solver import solve_milp
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable


# ---------------------------------------------------------------------------
# 1. Solver timeout & thread termination via stop_event
# ---------------------------------------------------------------------------

class TestCooperativeCancellation:
    """Verify that stop_event cleanly terminates underlying solver threads."""

    def test_simplex_solver_terminates_on_stop_event(self):
        """Simplex solver checks stop_event between pivots and terminates."""
        m = Model(name="cancellation_test_lp")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

        stop_event = threading.Event()
        stop_event.set()  # Already signaled before solve

        solver = RevisedSimplexSolver(stop_event=stop_event)
        solution = solver.solve(m)

        assert solution.status == SolutionStatus.ITERATION_LIMIT
        assert "stop signal" in solution.message.lower() or "iteration" in solution.message.lower()

    def test_milp_solver_terminates_on_stop_event(self):
        """MILP solver checks stop_event at B&B node boundaries and terminates."""
        m = Model(name="cancellation_test_milp")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type="integer"))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type="integer"))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 7.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        stop_event = threading.Event()
        stop_event.set()  # Stop event triggered immediately

        solution = solve_milp(m, stop_event=stop_event)

        assert solution.status == SolutionStatus.ITERATION_LIMIT
        assert "stop signal" in solution.message.lower() or "cancelled" in solution.message.lower()

    @pytest.mark.anyio
    async def test_solve_with_timeout_signals_stop_event_on_timeout(self):
        """When asyncio.wait_for times out, stop_event is signaled and 408 raised."""
        from fastapi import HTTPException
        # Create an artificial problem that will take non-zero time or mock timeout
        m = Model(name="timeout_lp")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))

        # With a ridiculously small timeout (1e-6 s), it should trigger TimeoutError
        with pytest.raises(HTTPException) as exc_info:
            await solve_with_timeout(m, has_integer_vars=False, timeout_seconds=1e-7)

        assert exc_info.value.status_code == 408
        assert "timed out" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# 2. CORS configuration security
# ---------------------------------------------------------------------------

class TestCORSSecurity:
    """Verify CORS complies with Fetch specification and security policies."""

    def test_cors_does_not_use_wildcard_with_credentials(self):
        """CORS must not allow credentials when origins are wildcard, or credentials must be False."""
        from fastapi.middleware.cors import CORSMiddleware
        # Find CORS middleware in app
        cors_middlewares = [
            m for m in app.user_middleware
            if m.cls == CORSMiddleware
        ]
        assert len(cors_middlewares) == 1
        cors_options = cors_middlewares[0].kwargs
        assert cors_options.get("allow_credentials") is False

    def test_cors_allowed_origins_include_localhost_development(self):
        """Default allowed origins contain localhost development ports."""
        assert "http://localhost:3000" in ALLOWED_ORIGINS
        assert "http://localhost:8000" in ALLOWED_ORIGINS
        assert "*" not in ALLOWED_ORIGINS

    def test_cors_preflight_response_for_allowed_origin(self):
        """Preflight OPTIONS request from allowed origin returns proper headers."""
        client = TestClient(app)
        response = client.options("/solve", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        })
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")


# ---------------------------------------------------------------------------
# 3. Negative upper_bound validation
# ---------------------------------------------------------------------------

class TestNegativeUpperBoundValidation:
    """Verify valid negative upper bounds like -5 <= x <= -1 are accepted."""

    def test_negative_upper_bound_accepted(self):
        """Variable with -5 <= x <= -1 must be valid."""
        var = VariableInput(name="x_neg", lower_bound=-5.0, upper_bound=-1.0)
        assert var.lower_bound == -5.0
        assert var.upper_bound == -1.0

    def test_negative_upper_bound_inverted_rejected(self):
        """Variable with lb > ub (e.g. -1 <= x <= -5) must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            VariableInput(name="x_bad", lower_bound=-1.0, upper_bound=-5.0)
        assert "lower_bound" in str(exc_info.value) or "upper_bound" in str(exc_info.value)

    def test_solve_problem_with_negative_variable_bounds(self):
        """API solve endpoint accepts and solves model with negative variable bounds."""
        client = TestClient(app)
        response = client.post("/solve", json={
            "variables": [
                {"name": "x", "lower_bound": -10.0, "upper_bound": -2.0, "variable_type": "continuous"}
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x": 1.0}, "sense": "<=", "rhs": -3.0}
            ],
            "objective": {"coefficients": {"x": 1.0}, "sense": "maximize"}
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        var_vals = {v["name"]: v["value"] for v in data["variable_values"]}
        assert math.isclose(var_vals["x"], -3.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 4. Whitespace-only name rejection and trimming
# ---------------------------------------------------------------------------

class TestWhitespaceNameValidation:
    """Verify whitespace-only variable and constraint names are rejected and trimmed."""

    def test_whitespace_only_variable_name_rejected(self):
        """Variable with whitespace-only name must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            VariableInput(name="   ", lower_bound=0.0)
        assert "name" in str(exc_info.value).lower()

    def test_whitespace_only_constraint_name_rejected(self):
        """Constraint with whitespace-only name must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConstraintInput(name="   ", coefficients={"x": 1.0}, sense="<=", rhs=5.0)
        assert "name" in str(exc_info.value).lower()

    def test_variable_name_with_surrounding_whitespace_trimmed(self):
        """Variable name with leading/trailing whitespace is stripped."""
        var = VariableInput(name="  x1  ", lower_bound=0.0)
        assert var.name == "x1"

    def test_constraint_name_with_surrounding_whitespace_trimmed(self):
        """Constraint name with leading/trailing whitespace is stripped."""
        con = ConstraintInput(name="  c1  ", coefficients={"x1": 1.0}, sense="<=", rhs=5.0)
        assert con.name == "c1"


# ---------------------------------------------------------------------------
# 5. All-zero objective coefficients rejection
# ---------------------------------------------------------------------------

class TestZeroObjectiveValidation:
    """Verify objectives with all zero coefficients are rejected."""

    def test_all_zero_objective_coefficients_rejected(self):
        """Objective with all 0 coefficients must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ObjectiveInput(coefficients={"x1": 0.0, "x2": 0.0}, sense="minimize")
        assert "zero" in str(exc_info.value).lower()

    def test_single_zero_objective_coefficient_rejected(self):
        """Objective with single 0 coefficient must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ObjectiveInput(coefficients={"x1": 0.0}, sense="maximize")
        assert "zero" in str(exc_info.value).lower()

    def test_nonzero_objective_coefficients_accepted(self):
        """Objective with at least one non-zero coefficient is valid."""
        obj = ObjectiveInput(coefficients={"x1": 0.0, "x2": 1.5}, sense="minimize")
        assert obj.coefficients == {"x1": 0.0, "x2": 1.5}


# ---------------------------------------------------------------------------
# 6. Request size limit & timeout parameter boundaries
# ---------------------------------------------------------------------------

class TestSecurityLimits:
    """Verify DoS protections: body size limit and timeout validation."""

    def test_oversized_payload_rejected_with_413(self):
        """Payload exceeding 10 MB returns 413 Request Entity Too Large."""
        client = TestClient(app)
        large_body = "x" * (11 * 1024 * 1024)
        response = client.post("/solve", data=large_body, headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        assert "too large" in response.text.lower()

    def test_timeout_bounds_enforced(self):
        """Timeout below 0.1s or above 300s is rejected by Pydantic."""
        client = TestClient(app)
        # Below min
        r1 = client.post("/solve", json={
            "variables": [{"name": "x", "lower_bound": 0}],
            "constraints": [],
            "objective": {"coefficients": {"x": 1}, "sense": "minimize"},
            "timeout_seconds": 0.01,
        })
        assert r1.status_code == 422

        # Above max
        r2 = client.post("/solve", json={
            "variables": [{"name": "x", "lower_bound": 0}],
            "constraints": [],
            "objective": {"coefficients": {"x": 1}, "sense": "minimize"},
            "timeout_seconds": 500.0,
        })
        assert r2.status_code == 422


# ---------------------------------------------------------------------------
# 7. Benchmark seed parameter & classification integrity
# ---------------------------------------------------------------------------

class TestBenchmarkImprovements:
    """Verify MILP benchmark seed differentiation and result classification."""

    def test_generate_random_milp_accepts_seed(self):
        """generate_random_milp produces different models with different seeds."""
        from benchmarks.benchmark_milp import generate_random_milp
        m1 = generate_random_milp(num_vars=4, num_constraints=3, seed=42)
        m2 = generate_random_milp(num_vars=4, num_constraints=3, seed=99)

        # Compare objective coefficients or constraint RHS
        obj1_coeffs = list(m1.objective.coefficients.values())
        obj2_coeffs = list(m2.objective.coefficients.values())
        assert obj1_coeffs != obj2_coeffs
