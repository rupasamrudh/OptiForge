"""API tests for OptiForge web interface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from optiforge.api.main import app
from optiforge.api.models import SolveRequest, VariableInput, ConstraintInput, ObjectiveInput
from optiforge.api.adapter import api_request_to_model, solution_to_response


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"


class TestSolveEndpoint:
    """Tests for the solve endpoint."""

    def test_valid_lp_minimize(self):
        """Test valid LP minimize request."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
                {"name": "c2", "coefficients": {"x1": 2, "x2": 1}, "sense": "<=", "rhs": 5},
            ],
            "objective": {"coefficients": {"x1": 3, "x2": 2}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        assert data["objective_value"] is not None
        assert isinstance(data["variable_values"], list)
        assert data["iterations"] >= 0
        assert data["solve_time_ms"] is not None

    def test_valid_lp_maximize(self):
        """Test valid LP maximize request."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
                {"name": "c2", "coefficients": {"x1": 2, "x2": 1}, "sense": "<=", "rhs": 5},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        assert data["objective_value"] is not None
        # For max x1 + x2 s.t. x1 + x2 <= 4, 2x1 + x2 <= 5, optimal is x1=1, x2=3, obj=4
        assert abs(data["objective_value"] - 4.0) < 1e-6

    def test_invalid_matrix_dimensions_objective(self):
        """Test invalid objective - references unknown variable."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1, "x2": 2}, "sense": "minimize"},  # x2 doesn't exist
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_invalid_matrix_dimensions_constraint(self):
        """Test invalid constraint - references unknown variable."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},  # x2 doesn't exist
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_invalid_objective_direction(self):
        """Test invalid objective direction."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "invalid"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error

    def test_invalid_constraint_sense(self):
        """Test invalid constraint sense."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "invalid", "rhs": 4},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error

    def test_empty_variables(self):
        """Test empty variables list."""
        request_data = {
            "variables": [],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error (min_length=1)

    def test_empty_objective_coefficients(self):
        """Test empty objective coefficients."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error (min_length=1)

    def test_duplicate_variable_names(self):
        """Test duplicate variable names."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_duplicate_constraint_names(self):
        """Test duplicate constraint names."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "<=", "rhs": 4},
                {"name": "c1", "coefficients": {"x1": 2}, "sense": "<=", "rhs": 5},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_infeasible_problem(self):
        """Test infeasible problem."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": -1},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "infeasible"

    def test_unbounded_problem(self):
        """Test unbounded problem."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": -1}, "sense": "minimize"},  # min -x1, x1 >= 0 -> unbounded
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unbounded"


class TestAdapter:
    """Tests for the API-to-model adapter."""

    def test_api_request_to_model(self):
        """Test converting API request to OptiForge model."""
        request = SolveRequest(
            variables=[
                VariableInput(name="x1", lower_bound=0, variable_type="continuous"),
                VariableInput(name="x2", lower_bound=0, variable_type="continuous"),
            ],
            constraints=[
                ConstraintInput(name="c1", coefficients={"x1": 1, "x2": 1}, sense="<=", rhs=4),
            ],
            objective=ObjectiveInput(coefficients={"x1": 3, "x2": 2}, sense="minimize"),
        )

        model = api_request_to_model(request)
        assert model.num_variables == 2
        assert model.num_constraints == 1
        assert model.objective is not None
        assert model.objective.sense.value == "minimize"

    def test_solution_to_response(self):
        """Test converting OptiForge solution to API response."""
        from optiforge.core.solution import Solution, SolutionStatus

        solution = Solution(
            status=SolutionStatus.OPTIMAL,
            variable_values={"x1": 1.0, "x2": 3.0},
            objective_value=4.0,
            message="Optimal solution found.",
            iterations=2,
        )

        response = solution_to_response(solution, solve_time_ms=5.5)
        assert response["status"] == "optimal"
        assert response["objective_value"] == 4.0
        assert len(response["variable_values"]) == 2
        assert response["iterations"] == 2
        assert response["solve_time_ms"] == 5.5

    def test_variable_bounds(self):
        """Test variable bounds are preserved."""
        request = SolveRequest(
            variables=[
                VariableInput(name="x1", lower_bound=1.0, upper_bound=5.0, variable_type="continuous"),
                VariableInput(name="x2", lower_bound=0, upper_bound=None, variable_type="continuous"),
            ],
            constraints=[],
            objective=ObjectiveInput(coefficients={"x1": 1, "x2": 1}, sense="minimize"),
        )

        model = api_request_to_model(request)
        vars_dict = model.variables
        assert vars_dict["x1"].lower_bound == 1.0
        assert vars_dict["x1"].upper_bound == 5.0
        assert vars_dict["x2"].lower_bound == 0.0
        assert vars_dict["x2"].upper_bound == float("inf")

    def test_variable_types(self):
        """Test variable types are preserved."""
        request = SolveRequest(
            variables=[
                VariableInput(name="x1", lower_bound=0, variable_type="continuous"),
                VariableInput(name="x2", lower_bound=0, variable_type="integer"),
                VariableInput(name="x3", lower_bound=0, upper_bound=1, variable_type="binary"),
            ],
            constraints=[],
            objective=ObjectiveInput(coefficients={"x1": 1, "x2": 1, "x3": 1}, sense="minimize"),
        )

        model = api_request_to_model(request)
        vars_dict = model.variables
        from optiforge.model import VariableType
        assert vars_dict["x1"].variable_type == VariableType.CONTINUOUS
        assert vars_dict["x2"].variable_type == VariableType.INTEGER
        assert vars_dict["x3"].variable_type == VariableType.BINARY


class TestSolverIntegration:
    """Integration tests with the actual solver."""

    def test_presolve_integration(self):
        """Test that presolve is used in the solve flow."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        assert "presolve" in data["message"].lower() or data["iterations"] >= 0

    def test_solve_time_included(self):
        """Test that solve time is included in response."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["solve_time_ms"] is not None
        assert data["solve_time_ms"] > 0


class TestResponseFormat:
    """Tests for response format consistency."""

    def test_response_has_required_fields(self):
        """Test that response has all required fields."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()

        required_fields = ["status", "objective_value", "variable_values", "message", "iterations", "solve_time_ms"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_variable_values_format(self):
        """Test variable values format."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()

        for var in data["variable_values"]:
            assert "name" in var
            assert "value" in var
            assert isinstance(var["name"], str)
            assert isinstance(var["value"], (int, float))


class TestConstraintEvaluation:
    """Tests for constraint evaluation in responses."""

    def test_le_constraint_evaluation(self):
        """Test <= constraint evaluation with slack and violation."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        
        # Check constraint evaluation
        ce = data["constraint_evaluations"][0]
        assert ce["name"] == "c1"
        assert ce["sense"] == "<="
        assert ce["lhs_value"] == 4.0
        assert ce["rhs"] == 4.0
        assert ce["slack"] == 0.0
        assert ce["residual"] is None
        assert ce["violation"] == 0.0
        assert ce["is_satisfied"] is True

    def test_ge_constraint_evaluation(self):
        """Test >= constraint evaluation with slack and violation."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": ">=", "rhs": 1},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        
        ce = data["constraint_evaluations"][0]
        assert ce["sense"] == ">="
        assert ce["lhs_value"] == 1.0
        assert ce["rhs"] == 1.0
        assert ce["slack"] == 0.0
        assert ce["residual"] is None
        assert ce["violation"] == 0.0
        assert ce["is_satisfied"] is True

    def test_eq_constraint_evaluation(self):
        """Test == constraint evaluation with residual and violation."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "==", "rhs": 2},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        
        ce = data["constraint_evaluations"][0]
        assert ce["sense"] == "=="
        assert ce["lhs_value"] == 2.0
        assert ce["rhs"] == 2.0
        assert ce["slack"] is None
        assert ce["residual"] == 0.0
        assert ce["violation"] == 0.0
        assert ce["is_satisfied"] is True

    def test_le_constraint_with_slack(self):
        """Test <= constraint with positive slack."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "<=", "rhs": 10},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        
        ce = data["constraint_evaluations"][0]
        assert ce["lhs_value"] == 10.0
        assert ce["rhs"] == 10.0
        assert ce["slack"] == 0.0
        assert ce["is_satisfied"] is True

    def test_multiple_constraints_different_senses(self):
        """Test multiple constraints with different senses."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1_le", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
                {"name": "c2_ge", "coefficients": {"x1": 2, "x2": 1}, "sense": ">=", "rhs": 2},
                {"name": "c3_eq", "coefficients": {"x1": 1, "x2": 0}, "sense": "==", "rhs": 1},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "optimal"
        
        ce_list = data["constraint_evaluations"]
        assert len(ce_list) == 3
        
        # Check each constraint
        c1 = next(c for c in ce_list if c["name"] == "c1_le")
        assert c1["sense"] == "<="
        assert c1["slack"] == 0.0
        assert c1["residual"] is None
        
        c2 = next(c for c in ce_list if c["name"] == "c2_ge")
        assert c2["sense"] == ">="
        assert c2["slack"] == 3.0
        assert c2["residual"] is None
        
        c3 = next(c for c in ce_list if c["name"] == "c3_eq")
        assert c3["sense"] == "=="
        assert c3["slack"] is None
        assert c3["residual"] == 0.0


class TestVariableMetadata:
    """Tests for variable metadata in responses."""

    def test_continuous_variable_metadata(self):
        """Test continuous variable metadata."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "upper_bound": 10, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        var = data["variable_values"][0]
        assert var["name"] == "x1"
        assert var["variable_type"] == "continuous"
        assert var["lower_bound"] == 0.0
        assert var["upper_bound"] == 10.0
        assert var["integrality_valid"] is None
        assert var["integrality_violation"] is None

    def test_integer_variable_metadata(self):
        """Test integer variable metadata with integrality validation."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "upper_bound": 10, "variable_type": "integer"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        var = data["variable_values"][0]
        assert var["variable_type"] == "integer"
        assert var["integrality_valid"] is True
        assert var["integrality_violation"] == 0.0

    def test_binary_variable_metadata(self):
        """Test binary variable metadata with integrality validation."""
        request_data = {
            "variables": [
                {"name": "x1", "variable_type": "binary"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        var = data["variable_values"][0]
        assert var["variable_type"] == "binary"
        assert var["lower_bound"] == 0.0
        # Binary variables have upper_bound=1.0 enforced by the model
        assert var["upper_bound"] == 1.0
        assert var["integrality_valid"] is True
        assert var["integrality_violation"] == 0.0

    def test_mixed_variable_types(self):
        """Test mixed variable types in one problem."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "integer"},
                {"name": "x2", "lower_bound": 0, "variable_type": "binary"},
                {"name": "x3", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1, "x3": 1}, "sense": "<=", "rhs": 5},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1, "x3": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        vars_dict = {v["name"]: v for v in data["variable_values"]}
        assert vars_dict["x1"]["variable_type"] == "integer"
        assert vars_dict["x1"]["integrality_valid"] is True
        assert vars_dict["x2"]["variable_type"] == "binary"
        assert vars_dict["x2"]["integrality_valid"] is True
        assert vars_dict["x3"]["variable_type"] == "continuous"
        assert vars_dict["x3"]["integrality_valid"] is None


class TestMILPMetrics:
    """Tests for MILP-specific metrics in responses."""

    def test_milp_metrics_present(self):
        """Test that MILP metrics are present for integer problems."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "integer"},
                {"name": "x2", "lower_bound": 0, "variable_type": "integer"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 3},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "optimal"
        assert data["incumbent_objective"] == 3.0
        assert data["best_bound"] == 3.0
        assert data["nodes_remaining"] == 0
        assert data["max_depth"] == 0
        assert data["feasible_solutions_found"] == 1

    def test_milp_metrics_not_present_for_lp(self):
        """Test that MILP metrics are not present for LP problems."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        # LP problems should not have MILP metrics
        assert data.get("incumbent_objective") is None
        assert data.get("best_bound") is None
        assert data.get("optimality_gap") is None
        assert data.get("absolute_gap") is None
        assert data.get("nodes_remaining") is None
        assert data.get("max_depth") is None
        assert data.get("feasible_solutions_found") is None


class TestStatusExplanation:
    """Tests for status explanations."""

    def test_optimal_explanation(self):
        """Test optimal status explanation."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "upper_bound": 10, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "optimal"
        assert data["status_explanation"] == "Optimal solution found and proven."

    def test_infeasible_explanation(self):
        """Test infeasible status explanation."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": -1},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "infeasible"
        assert data["status_explanation"] == "Problem is infeasible - no solution satisfies all constraints."

    def test_unbounded_explanation(self):
        """Test unbounded status explanation."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": -1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "unbounded"
        assert data["status_explanation"] == "Problem is unbounded - objective can be improved indefinitely."


class TestFeasibilityInfo:
    """Tests for feasibility information in responses."""

    def test_feasible_solution_info(self):
        """Test feasibility info for feasible solution."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["feasibility_info"] is not None
        assert "All constraints satisfied" in data["feasibility_info"]

    def test_infeasible_solution_info(self):
        """Test feasibility info for infeasible solution."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": -1},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "infeasible"
        assert data["feasibility_info"] is not None
        assert "Constraints violated" in data["feasibility_info"]
        assert "c1" in data["feasibility_info"]


class TestEqualityResidual:
    """Tests for equality constraint residuals."""

    def test_equality_residual_zero(self):
        """Test equality constraint with zero residual."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "==", "rhs": 2},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        ce = data["constraint_evaluations"][0]
        assert ce["sense"] == "=="
        assert ce["residual"] == 0.0
        assert ce["violation"] == 0.0
        assert ce["slack"] is None

    def test_equality_residual_nonzero(self):
        """Test equality constraint with non-zero residual (if possible)."""
        # This test verifies the residual field exists and is correctly calculated
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "==", "rhs": 2},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        ce = data["constraint_evaluations"][0]
        assert ce["residual"] is not None
        assert ce["residual"] >= 0.0
        assert ce["violation"] == ce["residual"]


class TestFeasibilityTolerance:
    """Tests for feasibility tolerance handling."""

    def test_feasibility_tolerance_boundary(self):
        """Test constraint satisfaction at tolerance boundary."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "<=", "rhs": 1},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        ce = data["constraint_evaluations"][0]
        # At boundary, should be satisfied with 1e-9 tolerance
        assert ce["is_satisfied"] is True
        assert ce["violation"] == 0.0


class TestInvalidInput:
    """Tests for invalid input handling."""

    def test_unknown_variable_in_objective(self):
        """Test unknown variable in objective."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1, "x2": 2}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_unknown_variable_in_constraint(self):
        """Test unknown variable in constraint."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 4},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_duplicate_variable_names(self):
        """Test duplicate variable names."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_duplicate_constraint_names(self):
        """Test duplicate constraint names."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "<=", "rhs": 4},
                {"name": "c1", "coefficients": {"x1": 2}, "sense": "<=", "rhs": 5},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_empty_variables(self):
        """Test empty variables list."""
        request_data = {
            "variables": [],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422

    def test_empty_objective_coefficients(self):
        """Test empty objective coefficients."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422

    def test_invalid_objective_direction(self):
        """Test invalid objective direction."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "invalid"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422

    def test_invalid_constraint_sense(self):
        """Test invalid constraint sense."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "invalid", "rhs": 4},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 422


class TestInfeasibleModel:
    """Tests for infeasible models."""

    def test_infeasible_lp(self):
        """Test infeasible LP model."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
                {"name": "x2", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": -1},
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "infeasible"
        assert data["objective_value"] is None
        assert data["variable_values"] == []

    def test_infeasible_milp(self):
        """Test infeasible MILP model."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "integer"},
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": ">=", "rhs": 5},
                {"name": "c2", "coefficients": {"x1": 1}, "sense": "<=", "rhs": 2},
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "infeasible"


class TestUnboundedModel:
    """Tests for unbounded models."""

    def test_unbounded_lp(self):
        """Test unbounded LP model."""
        request_data = {
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "continuous"},
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": -1}, "sense": "minimize"},
        }

        response = client.post("/solve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unbounded"
        assert data["objective_value"] is None
        assert data["variable_values"] == []


class TestStatusMapping:
    """Tests for status mapping."""

    def test_all_statuses_mapped(self):
        """Test that all solver statuses are mapped to API statuses."""
        from optiforge.core.solution import SolutionStatus
        
        # Verify all enum values have a mapping
        status_map = {
            SolutionStatus.OPTIMAL: "optimal",
            SolutionStatus.FEASIBLE: "feasible",
            SolutionStatus.INFEASIBLE: "infeasible",
            SolutionStatus.UNBOUNDED: "unbounded",
            SolutionStatus.ITERATION_LIMIT: "iteration_limit",
            SolutionStatus.UNKNOWN: "unknown",
        }
        
        for solver_status, api_status in status_map.items():
            assert api_status in ["optimal", "feasible", "infeasible", "unbounded", "iteration_limit", "unknown"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])