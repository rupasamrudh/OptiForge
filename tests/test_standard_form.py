"""Unit tests for LP standard-form conversion and Phase-I preparation (Stage 2.5)."""

import math
import numpy as np
import pytest

from optiforge.model import (
    Variable,
    VariableType,
    Constraint,
    ConstraintSense,
    Objective,
    ObjectiveSense,
    Model,
)
from optiforge.lp import (
    SingularBasisError,
    StandardFormModel,
    VariableInfo,
    VariableRole,
    to_standard_form,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _build_basic_lp() -> Model:
    """Build the canonical 2-variable LP used throughout the tests.

    maximize   3·x1 + 2·x2
    s.t.
        x1 +  x2 <= 4
       2·x1 + x2 <= 5
        x1, x2 >= 0
    """
    m = Model(name="basic_lp")
    m.add_variable(Variable(name="x1", lower_bound=0.0))
    m.add_variable(Variable(name="x2", lower_bound=0.0))
    m.add_constraint(
        Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0},
                   sense=ConstraintSense.LESS_EQUAL, rhs=4.0)
    )
    m.add_constraint(
        Constraint(name="c2", coefficients={"x1": 2.0, "x2": 1.0},
                   sense=ConstraintSense.LESS_EQUAL, rhs=5.0)
    )
    m.set_objective(
        Objective(coefficients={"x1": 3.0, "x2": 2.0},
                  sense=ObjectiveSense.MAXIMIZE)
    )
    return m


def _build_mixed_lp() -> Model:
    """Build an LP with <=, >=, and == constraints.

    minimize   x1 + 2·x2 + 3·x3
    s.t.
        x1 + x2       <= 10    (row 0: slack s1)
             x2 + x3  >= 3     (row 1: surplus e2, artificial a2)
        x1      + x3  ==  5    (row 2: artificial a3)
        x1, x2, x3 >= 0
    """
    m = Model(name="mixed_lp")
    m.add_variable(Variable(name="x1", lower_bound=0.0))
    m.add_variable(Variable(name="x2", lower_bound=0.0))
    m.add_variable(Variable(name="x3", lower_bound=0.0))
    m.add_constraint(
        Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0},
                   sense=ConstraintSense.LESS_EQUAL, rhs=10.0)
    )
    m.add_constraint(
        Constraint(name="c2", coefficients={"x2": 1.0, "x3": 1.0},
                   sense=ConstraintSense.GREATER_EQUAL, rhs=3.0)
    )
    m.add_constraint(
        Constraint(name="c3", coefficients={"x1": 1.0, "x3": 1.0},
                   sense=ConstraintSense.EQUAL, rhs=5.0)
    )
    m.set_objective(
        Objective(coefficients={"x1": 1.0, "x2": 2.0, "x3": 3.0},
                  sense=ObjectiveSense.MINIMIZE)
    )
    return m


# ── Slack variable tests ───────────────────────────────────────────────────


class TestSlackVariables:
    """<= constraints must create slack variables."""

    def test_le_creates_slack(self):
        sf = to_standard_form(_build_basic_lp())
        slack_info = [v for v in sf.variable_info if v.role is VariableRole.SLACK]
        assert len(slack_info) == 2
        assert slack_info[0].name == "s1"
        assert slack_info[1].name == "s2"

    def test_slack_coefficient_is_plus_one(self):
        sf = to_standard_form(_build_basic_lp())
        assert sf.A[0, 2] == 1.0
        assert sf.A[1, 3] == 1.0
        assert sf.A[1, 2] == 0.0
        assert sf.A[0, 3] == 0.0


# ── Surplus variable tests ─────────────────────────────────────────────────


class TestSurplusVariables:
    """>= constraints must create surplus variables."""

    def test_ge_creates_surplus(self):
        sf = to_standard_form(_build_mixed_lp())
        surplus_info = [v for v in sf.variable_info if v.role is VariableRole.SURPLUS]
        assert len(surplus_info) == 1
        assert surplus_info[0].name == "e2"

    def test_surplus_coefficient_is_minus_one(self):
        sf = to_standard_form(_build_mixed_lp())
        surplus_col = next(
            v.column_index for v in sf.variable_info
            if v.role is VariableRole.SURPLUS
        )
        assert sf.A[1, surplus_col] == -1.0


# ── Equality & Artificial variable tests ───────────────────────────────────


class TestEqualityConstraints:
    """== constraints create an artificial variable for Phase-I starting basis."""

    def test_eq_creates_artificial_not_slack_or_surplus(self):
        sf = to_standard_form(_build_mixed_lp())
        # In mixed LP: 1 slack (<=), 1 surplus (>=), 2 artificials (>= and ==).
        assert sf.num_slack_variables == 1
        assert sf.num_surplus_variables == 1
        assert sf.num_artificial_variables == 2
        # Total columns: 3 orig + 1 slack + 1 surplus + 2 artificial = 7
        assert sf.num_variables == 7


# ── A matrix tests ─────────────────────────────────────────────────────────


class TestAMatrix:
    """Correct A matrix for the basic LP."""

    def test_a_shape(self):
        sf = to_standard_form(_build_basic_lp())
        assert sf.A.shape == (2, 4)

    def test_a_values(self):
        sf = to_standard_form(_build_basic_lp())
        expected = np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ])
        np.testing.assert_array_equal(sf.A, expected)


# ── b vector tests ─────────────────────────────────────────────────────────


class TestBVector:
    def test_b_values(self):
        sf = to_standard_form(_build_basic_lp())
        np.testing.assert_array_equal(sf.b, np.array([4.0, 5.0]))


# ── Objective vector tests ─────────────────────────────────────────────────


class TestObjectiveVector:
    def test_c_values(self):
        sf = to_standard_form(_build_basic_lp())
        np.testing.assert_array_equal(sf.c, np.array([3.0, 2.0, 0.0, 0.0]))

    def test_objective_sense_preserved(self):
        sf = to_standard_form(_build_basic_lp())
        assert sf.objective_sense is ObjectiveSense.MAXIMIZE

    def test_minimize_sense_preserved(self):
        sf = to_standard_form(_build_mixed_lp())
        assert sf.objective_sense is ObjectiveSense.MINIMIZE


# ── Basis / nonbasic tests ─────────────────────────────────────────────────


class TestBasis:
    """Slack variables from <= constraints form the initial identity basis."""

    def test_basis_indices(self):
        sf = to_standard_form(_build_basic_lp())
        assert sf.basis_indices == [2, 3]

    def test_nonbasic_indices(self):
        sf = to_standard_form(_build_basic_lp())
        assert sf.nonbasic_indices == [0, 1]

    def test_identity_in_basis_columns(self):
        sf = to_standard_form(_build_basic_lp())
        B = sf.A[:, sf.basis_indices]
        np.testing.assert_array_equal(B, np.eye(2))


# ── Initial BFS tests ──────────────────────────────────────────────────────


class TestInitialBFS:
    def test_bfs_values(self):
        sf = to_standard_form(_build_basic_lp())
        bfs = sf.compute_initial_bfs()
        assert bfs["x1"] == pytest.approx(0.0)
        assert bfs["x2"] == pytest.approx(0.0)
        assert bfs["s1"] == pytest.approx(4.0)
        assert bfs["s2"] == pytest.approx(5.0)

    def test_initial_objective_value(self):
        sf = to_standard_form(_build_basic_lp())
        bfs = sf.compute_initial_bfs()
        obj_val = sf.compute_objective_value(bfs)
        assert obj_val == pytest.approx(0.0)

    def test_bfs_requires_artificial_raises(self):
        """Models with >= or == constraints cannot compute Phase-II BFS directly."""
        sf = to_standard_form(_build_mixed_lp())
        with pytest.raises(ValueError, match="artificial"):
            sf.compute_initial_bfs()


# ── Multiple / mixed constraint tests ──────────────────────────────────────


class TestMixedConstraints:
    """Standard-form conversion with a mix of <=, >=, == constraints."""

    def test_mixed_a_shape(self):
        sf = to_standard_form(_build_mixed_lp())
        # 3 constraints × (3 orig + 1 slack + 1 surplus + 2 artificial) = 3 × 7
        assert sf.A.shape == (3, 7)

    def test_mixed_variable_names(self):
        sf = to_standard_form(_build_mixed_lp())
        assert sf.variable_names == ["x1", "x2", "x3", "s1", "e2", "a2", "a3"]

    def test_mixed_requires_artificial(self):
        sf = to_standard_form(_build_mixed_lp())
        assert sf.requires_artificial == [1, 2]
        assert sf.needs_phase1 is True

    def test_mixed_basis_is_complete(self):
        sf = to_standard_form(_build_mixed_lp())
        # Complete m=3 basis: s1 (col 3), a2 (col 5), a3 (col 6)
        assert sf.phase1_basis_indices == [3, 5, 6]
        assert sf.basis_indices == [3, 5, 6]
        # Basis matrix must be identity I_3
        B = sf.A[:, sf.phase1_basis_indices]
        np.testing.assert_array_equal(B, np.eye(3))

    def test_mixed_a_values(self):
        sf = to_standard_form(_build_mixed_lp())
        expected = np.array([
            # x1   x2   x3   s1    e2   a2   a3
            [1.0, 1.0, 0.0, 1.0,  0.0, 0.0, 0.0],  # c1: x1 + x2 + s1 = 10
            [0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0],  # c2: x2 + x3 - e2 + a2 = 3
            [1.0, 0.0, 1.0, 0.0,  0.0, 0.0, 1.0],  # c3: x1 + x3 + a3 = 5
        ])
        np.testing.assert_array_equal(sf.A, expected)

    def test_mixed_b_values(self):
        sf = to_standard_form(_build_mixed_lp())
        np.testing.assert_array_equal(sf.b, np.array([10.0, 3.0, 5.0]))

    def test_mixed_c_values(self):
        sf = to_standard_form(_build_mixed_lp())
        # Original objective has 0 for slacks, surplus, and artificials
        np.testing.assert_array_equal(
            sf.c, np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        )


# ── Validation / error tests ──────────────────────────────────────────────


class TestValidation:
    def test_no_objective_raises(self):
        m = Model(name="no_obj")
        m.add_variable(Variable(name="x"))
        m.add_constraint(
            Constraint(name="c", coefficients={"x": 1.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=1.0)
        )
        with pytest.raises(ValueError, match="objective"):
            to_standard_form(m)

    def test_no_variables_raises(self):
        m = Model(name="empty")
        with pytest.raises(ValueError, match="variables"):
            to_standard_form(m)

    def test_integer_variable_raises(self):
        m = Model(name="int_var")
        m.add_variable(
            Variable(name="y", variable_type=VariableType.INTEGER,
                     lower_bound=0.0, upper_bound=5.0)
        )
        m.add_constraint(
            Constraint(name="c", coefficients={"y": 1.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=5.0)
        )
        m.set_objective(
            Objective(coefficients={"y": 1.0}, sense=ObjectiveSense.MINIMIZE)
        )
        with pytest.raises(ValueError, match="continuous"):
            to_standard_form(m)

    def test_nonzero_lower_bound_shifted(self):
        """Non-zero lower bounds are handled via variable shift x = y + L."""
        m = Model(name="lb_shift")
        m.add_variable(Variable(name="x", lower_bound=2.0, upper_bound=10.0))
        m.add_constraint(
            Constraint(name="c", coefficients={"x": 1.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=10.0)
        )
        m.set_objective(
            Objective(coefficients={"x": 1.0}, sense=ObjectiveSense.MINIMIZE)
        )
        sf = to_standard_form(m)
        # x = y + 2, y >= 0; constraint becomes y <= 8; objective constant = 2.
        assert 0 in sf.variable_shifts
        assert sf.variable_shifts[0] == 2.0
        assert sf.objective_constant == pytest.approx(2.0)
        # Upper bound row: y <= 8 (10 - 2)
        assert sf.has_upper_bounds is True

    def test_free_variable_raises(self):
        m = Model(name="free_var")
        m.add_variable(Variable(name="x", lower_bound=-math.inf))
        m.add_constraint(
            Constraint(name="c", coefficients={"x": 1.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=1.0)
        )
        m.set_objective(
            Objective(coefficients={"x": 1.0}, sense=ObjectiveSense.MINIMIZE)
        )
        with pytest.raises(ValueError, match="Free variable"):
            to_standard_form(m)

    def test_finite_upper_bound_recorded(self):
        m = Model(name="ub")
        m.add_variable(Variable(name="x", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(
            Constraint(name="c", coefficients={"x": 1.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=10.0)
        )
        m.set_objective(
            Objective(coefficients={"x": 1.0}, sense=ObjectiveSense.MINIMIZE)
        )
        sf = to_standard_form(m)
        assert 0 in sf.upper_bounds
        assert sf.upper_bounds[0] == 10.0
        assert sf.has_upper_bounds is True


# ── Three <= constraints (larger example) ──────────────────────────────────


class TestThreeConstraints:
    """Verify standard form for a model with 3 <= constraints."""

    @staticmethod
    def _build() -> Model:
        m = Model(name="three_le")
        m.add_variable(Variable(name="x1"))
        m.add_variable(Variable(name="x2"))
        m.add_variable(Variable(name="x3"))
        m.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0, "x3": 1.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=10.0)
        )
        m.add_constraint(
            Constraint(name="c2", coefficients={"x1": 2.0, "x3": 3.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=20.0)
        )
        m.add_constraint(
            Constraint(name="c3", coefficients={"x2": 4.0},
                       sense=ConstraintSense.LESS_EQUAL, rhs=12.0)
        )
        m.set_objective(
            Objective(coefficients={"x1": 5.0, "x2": 4.0, "x3": 3.0},
                       sense=ObjectiveSense.MAXIMIZE)
        )
        return m

    def test_a_shape(self):
        sf = to_standard_form(self._build())
        assert sf.A.shape == (3, 6)  # 3 orig + 3 slack

    def test_identity_basis(self):
        sf = to_standard_form(self._build())
        B = sf.A[:, sf.basis_indices]
        np.testing.assert_array_equal(B, np.eye(3))

    def test_bfs(self):
        sf = to_standard_form(self._build())
        bfs = sf.compute_initial_bfs()
        assert bfs["x1"] == pytest.approx(0.0)
        assert bfs["x2"] == pytest.approx(0.0)
        assert bfs["x3"] == pytest.approx(0.0)
        assert bfs["s1"] == pytest.approx(10.0)
        assert bfs["s2"] == pytest.approx(20.0)
        assert bfs["s3"] == pytest.approx(12.0)

    def test_initial_objective(self):
        sf = to_standard_form(self._build())
        bfs = sf.compute_initial_bfs()
        assert sf.compute_objective_value(bfs) == pytest.approx(0.0)


# ── STAGE 2.5: Negative RHS Normalization Tests ────────────────────────────


class TestNegativeRHSNormalization:
    """Every constraint must normalize to b_i >= 0."""

    def test_negative_rhs_le(self):
        """x1 + x2 <= -4  -->  -x1 - x2 >= 4  -->  -x1 - x2 - e1 + a1 = 4."""
        m = Model("neg_le")
        m.add_variable(Variable("x1"))
        m.add_variable(Variable("x2"))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, -4.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        sf = to_standard_form(m)
        assert sf.b[0] == 4.0
        # Row became >=, so it generated a surplus (e1) and artificial (a1)
        assert sf.num_slack_variables == 0
        assert sf.num_surplus_variables == 1
        assert sf.num_artificial_variables == 1
        # Coefficients of original variables negated
        assert sf.A[0, 0] == -1.0
        assert sf.A[0, 1] == -1.0
        assert sf.A[0, sf.surplus_indices[0]] == -1.0
        assert sf.A[0, sf.artificial_indices[0]] == 1.0
        assert sf.needs_phase1 is True

    def test_negative_rhs_ge(self):
        """x1 + 2·x2 >= -5  -->  -x1 - 2·x2 <= 5  -->  -x1 - 2·x2 + s1 = 5."""
        m = Model("neg_ge")
        m.add_variable(Variable("x1"))
        m.add_variable(Variable("x2"))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.GREATER_EQUAL, -5.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

        sf = to_standard_form(m)
        assert sf.b[0] == 5.0
        # Row became <=, so it generated a slack variable with +1
        assert sf.num_slack_variables == 1
        assert sf.num_surplus_variables == 0
        assert sf.num_artificial_variables == 0
        assert sf.A[0, 0] == -1.0
        assert sf.A[0, 1] == -2.0
        assert sf.A[0, sf.slack_indices[0]] == 1.0
        assert sf.needs_phase1 is False
        # Feasible starting BFS directly available!
        bfs = sf.compute_initial_bfs()
        assert bfs["s1"] == pytest.approx(5.0)

    def test_negative_rhs_eq(self):
        """2·x1 - x2 == -3  -->  -2·x1 + x2 == 3  -->  -2·x1 + x2 + a1 = 3."""
        m = Model("neg_eq")
        m.add_variable(Variable("x1"))
        m.add_variable(Variable("x2"))
        m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": -1.0}, ConstraintSense.EQUAL, -3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

        sf = to_standard_form(m)
        assert sf.b[0] == 3.0
        assert sf.A[0, 0] == -2.0
        assert sf.A[0, 1] == 1.0
        assert sf.num_artificial_variables == 1
        assert sf.A[0, sf.artificial_indices[0]] == 1.0
        assert sf.needs_phase1 is True


# ── STAGE 2.5: Artificial Variables & Phase-I Objective Tests ──────────────


class TestPhase1ObjectiveAndArtificials:
    """Verify artificial variable allocation and Phase-I objective."""

    def test_artificial_variable_metadata(self):
        m = Model("art_meta")
        m.add_variable(Variable("x1"))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.EQUAL, 7.0))
        m.set_objective(Objective({"x1": 5.0}, ObjectiveSense.MAXIMIZE))

        sf = to_standard_form(m)
        art_idx = sf.artificial_indices[0]
        v_info = sf.variable_info[art_idx]
        assert v_info.name == "a1"
        assert v_info.role is VariableRole.ARTIFICIAL
        assert v_info.column_index == art_idx

    def test_phase1_objective_vector(self):
        """Phase-I objective has 1.0 for artificials and 0.0 for everything else."""
        sf = to_standard_form(_build_mixed_lp())
        # Mixed LP has 7 variables: x1, x2, x3, s1, e2, a2, a3
        # a2 (col 5) and a3 (col 6) must have coefficient 1.0
        expected_phase1_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(sf.phase1_c, expected_phase1_c)
        assert sf.phase1_objective_sense is ObjectiveSense.MINIMIZE

        # Original objective remains completely intact
        expected_orig_c = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(sf.c, expected_orig_c)


# ── STAGE 2.5: Complete Basis Invariant & Validation Tests ─────────────────


class TestBasisValidationAndInvariant:
    """Verify that every system produces a full m-column basis."""

    def test_complete_basis_length_equals_m(self):
        sf = to_standard_form(_build_mixed_lp())
        assert len(sf.phase1_basis_indices) == sf.num_constraints
        assert len(sf.basis_indices) == sf.num_constraints

    def test_basis_matrix_is_identity(self):
        sf = to_standard_form(_build_mixed_lp())
        B = sf.A[:, sf.phase1_basis_indices]
        np.testing.assert_array_equal(B, np.eye(3))

    def test_validate_basis_passes_on_valid(self):
        sf = to_standard_form(_build_mixed_lp())
        sf.validate_basis()  # Should not raise

    def test_validate_basis_wrong_length(self):
        sf = to_standard_form(_build_mixed_lp())
        with pytest.raises(ValueError, match="exactly m columns"):
            sf.validate_basis([3, 5])  # 2 indices for 3 constraints

    def test_validate_basis_out_of_bounds(self):
        sf = to_standard_form(_build_mixed_lp())
        with pytest.raises(ValueError, match="out of bounds"):
            sf.validate_basis([0, 1, 99])

    def test_validate_basis_duplicate_indices(self):
        sf = to_standard_form(_build_mixed_lp())
        with pytest.raises(ValueError, match="unique"):
            sf.validate_basis([3, 5, 5])


# ── STAGE 2.5: Phase-I BFS Computation & Singular Basis Tests ──────────────


class TestPhase1BFS:
    """Verify initial Phase-I basic feasible solution computation."""

    def test_compute_phase1_bfs_mixed(self):
        sf = to_standard_form(_build_mixed_lp())
        bfs = sf.compute_phase1_bfs()
        assert bfs["x1"] == pytest.approx(0.0)
        assert bfs["x2"] == pytest.approx(0.0)
        assert bfs["x3"] == pytest.approx(0.0)
        assert bfs["s1"] == pytest.approx(10.0)
        assert bfs["e2"] == pytest.approx(0.0)
        assert bfs["a2"] == pytest.approx(3.0)
        assert bfs["a3"] == pytest.approx(5.0)

        # Phase-I objective value = a2 + a3 = 3 + 5 = 8.0
        w = sf.compute_phase1_objective_value(bfs)
        assert w == pytest.approx(8.0)

    def test_singular_basis_error(self):
        """A singular basis must raise SingularBasisError."""
        sf = to_standard_form(_build_basic_lp())
        # Column 0 is [1, 2]^T, column 1 is [1, 1]^T.
        # But let's build a linearly dependent sub-basis:
        # A has cols 2 (s1=[1,0]^T) and 3 (s2=[0,1]^T).
        # Replace basis with two identical columns, or manually construct singular basis.
        # Let's test _solve_basis_system with linearly dependent columns:
        # For a 2x4 system: create a model where two columns are linearly dependent.
        m = Model("singular")
        m.add_variable(Variable("x1"))
        m.add_variable(Variable("x2"))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 4.0))
        m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 4.0}, ConstraintSense.LESS_EQUAL, 8.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        sf_sing = to_standard_form(m)
        # Columns 0 and 1 are [1, 2]^T and [2, 4]^T -> linearly dependent!
        with pytest.raises(SingularBasisError, match="singular"):
            sf_sing._solve_basis_system([0, 1])


# ── STAGE 2.5: Three Specific Mathematical Examples ────────────────────────


class TestThreeMathematicalExamples:
    """Explicitly verify the three benchmark examples required by specification."""

    def test_example_1(self):
        """EXAMPLE 1:
        maximize: 3x1 + 2x2
        s.t.
            x1 + x2 <= 4
           2x1 + x2 <= 5
            x1, x2 >= 0
        Expected Phase-I basis: [s1, s2]
        Initial BFS: x1=0, x2=0, s1=4, s2=5, obj=0
        """
        sf = to_standard_form(_build_basic_lp())
        assert sf.needs_phase1 is False
        assert sf.phase1_basis_indices == [2, 3]
        B = sf.A[:, sf.phase1_basis_indices]
        np.testing.assert_array_equal(B, np.eye(2))

        bfs = sf.compute_initial_bfs()
        assert bfs["x1"] == pytest.approx(0.0)
        assert bfs["x2"] == pytest.approx(0.0)
        assert bfs["s1"] == pytest.approx(4.0)
        assert bfs["s2"] == pytest.approx(5.0)
        assert sf.compute_objective_value(bfs) == pytest.approx(0.0)

    def test_example_2(self):
        """EXAMPLE 2:
        minimize: x1 + 2x2
        s.t.
            x1 +  x2 >= 3
            x1 + 3x2 >= 6
            x1, x2 >= 0
        Expected augmented equations:
            x1 +  x2 - e1 + a1 = 3
            x1 + 3x2 - e2 + a2 = 6
        Expected Phase-I basis: [a1, a2]
        Initial BFS: a1=3, a2=6, Phase-I obj = 9
        """
        m = Model("ex2")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
        m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 3.0}, ConstraintSense.GREATER_EQUAL, 6.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

        sf = to_standard_form(m)
        assert sf.needs_phase1 is True
        assert sf.variable_names == ["x1", "x2", "e1", "a1", "e2", "a2"]
        # Expected A:
        # [1, 1, -1, 1,  0, 0]
        # [1, 3,  0, 0, -1, 1]
        expected_A = np.array([
            [1.0, 1.0, -1.0, 1.0,  0.0, 0.0],
            [1.0, 3.0,  0.0, 0.0, -1.0, 1.0],
        ])
        np.testing.assert_array_equal(sf.A, expected_A)
        np.testing.assert_array_equal(sf.b, np.array([3.0, 6.0]))

        # Phase-I basis contains a1 (col 3) and a2 (col 5)
        assert sf.phase1_basis_indices == [3, 5]
        B = sf.A[:, sf.phase1_basis_indices]
        np.testing.assert_array_equal(B, np.eye(2))

        bfs = sf.compute_phase1_bfs()
        assert bfs["x1"] == pytest.approx(0.0)
        assert bfs["x2"] == pytest.approx(0.0)
        assert bfs["e1"] == pytest.approx(0.0)
        assert bfs["e2"] == pytest.approx(0.0)
        assert bfs["a1"] == pytest.approx(3.0)
        assert bfs["a2"] == pytest.approx(6.0)
        assert sf.compute_phase1_objective_value(bfs) == pytest.approx(9.0)

    def test_example_3(self):
        """EXAMPLE 3:
        maximize: 2x1 + x2
        s.t.
            x1 + x2 = 4
            x1 <= 3
            x1, x2 >= 0
        Expected augmented equations:
            x1 + x2 + a1 = 4
            x1 + s2 = 3
        Expected Phase-I basis: [a1, s2]
        Initial BFS: a1=4, s2=3, Phase-I obj = 4
        """
        m = Model("ex3")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 4.0))
        m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        sf = to_standard_form(m)
        assert sf.needs_phase1 is True
        assert sf.variable_names == ["x1", "x2", "a1", "s2"]
        # Expected A:
        # [1, 1, 1, 0]  (c1: x1 + x2 + a1 = 4)
        # [1, 0, 0, 1]  (c2: x1 + s2 = 3)
        expected_A = np.array([
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_array_equal(sf.A, expected_A)
        np.testing.assert_array_equal(sf.b, np.array([4.0, 3.0]))

        # Phase-I basis contains a1 (col 2) and s2 (col 3)
        assert sf.phase1_basis_indices == [2, 3]
        B = sf.A[:, sf.phase1_basis_indices]
        np.testing.assert_array_equal(B, np.eye(2))

        bfs = sf.compute_phase1_bfs()
        assert bfs["x1"] == pytest.approx(0.0)
        assert bfs["x2"] == pytest.approx(0.0)
        assert bfs["a1"] == pytest.approx(4.0)
        assert bfs["s2"] == pytest.approx(3.0)
        assert sf.compute_phase1_objective_value(bfs) == pytest.approx(4.0)
