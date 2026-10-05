"""Test suite for Stage 8.11 Profiling-Guided Performance Optimizations.

Validates:
1. BorderedBasisFactorization correctly flattens multi-level recursive chains.
2. Root factorization and levels chain are correctly established.
3. Equivalence of flattened 1D and 2D solves against fresh DenseLU.
4. Equivalence of flattened 1D and 2D transpose solves against fresh DenseLU.
5. IncrementalBasisFactorization fast-path on empty eta updates.
6. Fallback safety and residual tolerance checks under flattened structure.
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.milp.solver import MILPSolver
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.bordered_basis import BorderedBasisFactorization
from optiforge.numerics.incremental_basis import IncrementalBasisFactorization
from optiforge.numerics.lu import dense_lu_factor, lu_solve, lu_solve_transpose


class TestStage811Optimizations:
    """Correctness and regression protection for Stage 8.11 optimizations."""

    def test_flattened_structure_depth_chain(self):
        """Verify _root_factorization points to root and _levels collects rows."""
        rng = np.random.RandomState(42)
        m0 = 4
        B0 = rng.randn(m0, m0) + np.eye(m0) * 5.0
        fact0 = BasisFactorization(B0)

        fact_curr = fact0
        B_curr = B0
        for depth in range(1, 6):
            m_curr = fact_curr.m
            a_k = rng.randn(m_curr)
            B_next = np.block([[B_curr, np.zeros((m_curr, 1))], [a_k, 1.0]])
            bordered = BorderedBasisFactorization(fact_curr, a_k, B_next)
            assert bordered._root_factorization is fact0
            assert len(bordered._levels) == depth
            fact_curr = bordered
            B_curr = B_next

    def test_flattened_1d_and_2d_solve_equivalence(self):
        """Verify 1D and 2D solves match fresh DenseLU to machine precision."""
        rng = np.random.RandomState(123)
        m0 = 5
        B0 = rng.randn(m0, m0) + np.eye(m0) * 4.0
        fact0 = BasisFactorization(B0)

        fact_curr = fact0
        B_curr = B0
        for depth in range(4):
            m_curr = fact_curr.m
            a_k = rng.randn(m_curr)
            B_next = np.block([[B_curr, np.zeros((m_curr, 1))], [a_k, 1.0]])
            fact_curr = BorderedBasisFactorization(fact_curr, a_k, B_next)
            B_curr = B_next

        m_total = fact_curr.m
        lu_ref = dense_lu_factor(B_curr)

        # 1D test
        rhs_1d = rng.randn(m_total)
        x_1d = fact_curr.solve(rhs_1d)
        x_ref_1d = lu_solve(lu_ref, rhs_1d)
        np.testing.assert_allclose(x_1d, x_ref_1d, atol=1e-12)

        y_1d = fact_curr.solve_transpose(rhs_1d)
        y_ref_1d = lu_solve_transpose(lu_ref, rhs_1d)
        np.testing.assert_allclose(y_1d, y_ref_1d, atol=1e-12)

        # 2D test (k=3 RHS vectors)
        rhs_2d = rng.randn(m_total, 3)
        x_2d = fact_curr.solve(rhs_2d)
        x_ref_2d = lu_solve(lu_ref, rhs_2d)
        np.testing.assert_allclose(x_2d, x_ref_2d, atol=1e-12)

        y_2d = fact_curr.solve_transpose(rhs_2d)
        y_ref_2d = lu_solve_transpose(lu_ref, rhs_2d)
        np.testing.assert_allclose(y_2d, y_ref_2d, atol=1e-12)

    def test_pfi_zero_eta_fast_path(self):
        """Verify IncrementalBasisFactorization fast-path when no pivots have occurred."""
        rng = np.random.RandomState(99)
        dim = 6
        B = rng.randn(dim, dim) + np.eye(dim) * 5.0
        inc = IncrementalBasisFactorization(B)
        assert len(inc._etas) == 0

        rhs = rng.randn(dim)
        x = inc.solve(rhs)
        y = inc.solve_transpose(rhs)

        lu = dense_lu_factor(B)
        np.testing.assert_allclose(x, lu_solve(lu, rhs), atol=1e-12)
        np.testing.assert_allclose(y, lu_solve_transpose(lu, rhs), atol=1e-12)

    def test_flattened_fallback_safety(self):
        """Verify fallback triggers safely if residual check fails in flattened bordered."""
        class FailingFactorization:
            def __init__(self, dim):
                self.m = dim
            def solve(self, rhs):
                return np.zeros_like(rhs)
            def solve_transpose(self, rhs):
                return np.zeros_like(rhs)

        B = np.eye(3)
        b_fact = BorderedBasisFactorization(
            FailingFactorization(2),
            a_B=np.zeros(2),
            B=B,
            residual_tolerance=1e-7,
        )
        rhs = np.array([1.0, 2.0, 3.0])
        x = b_fact.solve(rhs, check_residual=True)
        assert b_fact.is_fallback is True
        np.testing.assert_allclose(x, rhs)
