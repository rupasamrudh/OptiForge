"""Tests for API security and robustness features."""

import pytest
from fastapi.testclient import TestClient
from optiforge.api.main import app


class TestAPISecurityHeaders:
    """Test security headers are present on responses."""

    def test_security_headers_on_health(self):
        """Health endpoint should have security headers."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "Content-Security-Policy" in response.headers

    def test_security_headers_on_solve(self):
        """Solve endpoint should have security headers."""
        client = TestClient(app)
        response = client.post("/solve", json={
            "variables": [{"name": "x1", "lower_bound": 0}],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"}
        })
        
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_cors_headers(self):
        """CORS headers should be present for preflight requests."""
        client = TestClient(app)
        response = client.options("/solve", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        })
        
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")


class TestRequestSizeLimit:
    """Test request size limiting."""

    def test_large_request_rejected(self):
        """Requests exceeding size limit should return 413."""
        client = TestClient(app)
        
        # Create a request larger than 10MB
        large_data = "x" * (11 * 1024 * 1024)
        response = client.post("/solve", data=large_data, headers={
            "Content-Type": "application/json"
        })
        
        assert response.status_code == 413
        assert "too large" in response.text.lower()

    def test_normal_request_accepted(self):
        """Normal sized requests should be accepted (may fail validation but not size)."""
        client = TestClient(app)
        
        normal_data = '{"variables": [], "constraints": [], "objective": {"coefficients": {}, "sense": "maximize"}}'
        response = client.post("/solve", data=normal_data, headers={
            "Content-Type": "application/json"
        })
        
        # Should not be 413 (may be 422 for validation error)
        assert response.status_code != 413


class TestSolveTimeout:
    """Test solve timeout functionality."""

    def test_timeout_parameter_accepted(self):
        """Timeout parameter should be accepted in request."""
        client = TestClient(app)
        
        response = client.post("/solve", json={
            "variables": [{"name": "x1", "lower_bound": 0}],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
            "timeout_seconds": 60.0
        })
        
        # Should not fail due to timeout parameter
        assert response.status_code != 422 or "timeout_seconds" not in response.text

    def test_timeout_validation_min(self):
        """Timeout should reject values below minimum."""
        client = TestClient(app)
        
        response = client.post("/solve", json={
            "variables": [{"name": "x1", "lower_bound": 0}],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
            "timeout_seconds": 0.05  # Below minimum of 0.1
        })
        
        assert response.status_code == 422

    def test_timeout_validation_max(self):
        """Timeout should reject values above maximum."""
        client = TestClient(app)
        
        response = client.post("/solve", json={
            "variables": [{"name": "x1", "lower_bound": 0}],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"},
            "timeout_seconds": 400.0  # Above maximum of 300
        })
        
        assert response.status_code == 422


class TestErrorClassification:
    """Test error classification in API responses."""

    def test_infeasible_problem_returns_200_with_status(self):
        """Infeasible problems should return 200 with status in response body."""
        client = TestClient(app)
        
        # Create an infeasible problem: x <= -1, x >= 0
        response = client.post("/solve", json={
            "variables": [{"name": "x1", "lower_bound": 0}],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1}, "sense": "<=", "rhs": -1}
            ],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "infeasible"
        assert "infeasible" in data["message"].lower()

    def test_unbounded_problem_returns_200_with_status(self):
        """Unbounded problems should return 200 with status in response body."""
        client = TestClient(app)
        
        # Create an unbounded problem: max x, no upper bound
        response = client.post("/solve", json={
            "variables": [{"name": "x1", "lower_bound": 0}],  # No upper bound
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unbounded"
        assert "unbounded" in data["message"].lower()

    def test_validation_error_returns_422(self):
        """Validation errors should return 422 (Pydantic validation)."""
        client = TestClient(app)
        
        # Missing required fields
        response = client.post("/solve", json={
            "variables": [],
            "constraints": [],
            "objective": {"coefficients": {}, "sense": "maximize"}
        })
        
        assert response.status_code == 422

    def test_duplicate_variable_names_rejected(self):
        """Duplicate variable names should be rejected with 422."""
        client = TestClient(app)
        
        response = client.post("/solve", json={
            "variables": [
                {"name": "x1", "lower_bound": 0},
                {"name": "x1", "lower_bound": 0}  # Duplicate
            ],
            "constraints": [],
            "objective": {"coefficients": {"x1": 1}, "sense": "maximize"}
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "duplicate" in str(data).lower() or "unique" in str(data).lower()


class TestMILPMetricsInResponse:
    """Test MILP-specific metrics are included in response."""

    def test_milp_response_includes_metrics(self):
        """MILP solve should include MILP metrics in response."""
        client = TestClient(app)
        
        response = client.post("/solve", json={
            "variables": [
                {"name": "x1", "lower_bound": 0, "variable_type": "integer"},
                {"name": "x2", "lower_bound": 0, "variable_type": "integer"}
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 3}
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check MILP metrics are present
        assert "node_count" in data
        assert "incumbent_objective" in data
        assert "best_bound" in data
        assert "optimality_gap" in data
        assert "absolute_gap" in data
        assert "nodes_remaining" in data
        assert "max_depth" in data
        assert "feasible_solutions_found" in data


class TestLPResponse:
    """Test LP response structure."""

    def test_lp_response_structure(self):
        """LP solve should return proper structure."""
        client = TestClient(app)
        
        response = client.post("/solve", json={
            "variables": [
                {"name": "x1", "lower_bound": 0},
                {"name": "x2", "lower_bound": 0}
            ],
            "constraints": [
                {"name": "c1", "coefficients": {"x1": 1, "x2": 1}, "sense": "<=", "rhs": 3}
            ],
            "objective": {"coefficients": {"x1": 1, "x2": 1}, "sense": "maximize"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "objective_value" in data
        assert "iterations" in data
        assert "solve_time_ms" in data
        assert "variable_values" in data
        assert "constraint_evaluations" in data
        assert "feasibility_info" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])