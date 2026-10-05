"""Numerical diagnostics for basis matrices and LU factorizations.

This module provides structures and calculation helpers to inspect the numerical
properties of basis matrices, including pivot magnitudes, pivot growth, and
condition numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from optiforge.numerics.lu import DenseLU


@dataclass
class NumericalDiagnostics:
    """Diagnostic metrics for a basis matrix and its factorization.

    Attributes
    ----------
    dimension : int
        Dimension m of the m-by-m square basis matrix.
    min_abs_pivot : float
        Minimum absolute value of diagonal pivots in U: min_k |U[k, k]|.
    max_abs_pivot : float
        Maximum absolute value of diagonal pivots in U: max_k |U[k, k]|.
    pivot_growth : float
        Pivot growth factor: max_{i, j} |U[i, j]| / max_{i, j} |B[i, j]|.
    condition_number : float
        Estimated or computed 2-norm condition number of B.
        inf if B is singular or computation fails.
    is_singular : bool
        True if the factorization encountered a singular or near-zero pivot.
    success : bool
        True if factorization succeeded without numerical breakdown.
    """

    dimension: int
    min_abs_pivot: float
    max_abs_pivot: float
    pivot_growth: float
    condition_number: float
    is_singular: bool
    success: bool


def compute_diagnostics(
    B: np.ndarray,
    lu: DenseLU | None = None,
    estimate_condition: bool = True,
) -> NumericalDiagnostics:
    """Compute numerical diagnostics for a square matrix B and its LU factors.

    Parameters
    ----------
    B : np.ndarray
        The (m, m) basis matrix.
    lu : DenseLU, optional
        Pre-computed LU factors. If None, pivot metrics are set to NaN / 0.
    estimate_condition : bool, optional
        Whether to compute the condition number via np.linalg.cond.
        Default is True.

    Returns
    -------
    NumericalDiagnostics
        Object containing dimension, pivot extremes, growth, and condition number.
    """
    B_arr = np.asarray(B, dtype=np.float64)
    m = int(B_arr.shape[0]) if B_arr.ndim >= 1 else 0

    # Condition number estimation (diagnostic only)
    cond_num = float("nan")
    if estimate_condition and m > 0 and B_arr.ndim == 2 and B_arr.shape[0] == B_arr.shape[1]:
        try:
            cond_num = float(np.linalg.cond(B_arr))
        except (np.linalg.LinAlgError, ValueError):
            cond_num = float("inf")

    if lu is None or m == 0:
        return NumericalDiagnostics(
            dimension=m,
            min_abs_pivot=0.0,
            max_abs_pivot=0.0,
            pivot_growth=0.0,
            condition_number=cond_num,
            is_singular=True,
            success=False,
        )

    # Pivot metrics from U
    diag_pivots = np.abs(np.diag(lu.U))
    min_pivot = float(np.min(diag_pivots)) if m > 0 else 0.0
    max_pivot = float(np.max(diag_pivots)) if m > 0 else 0.0

    # Pivot growth: max |U_ij| / max |B_ij|
    max_b = float(np.max(np.abs(B_arr))) if m > 0 else 0.0
    max_u = float(np.max(np.abs(lu.U))) if m > 0 else 0.0
    pivot_growth = (max_u / max_b) if max_b > 1e-15 else 1.0

    is_singular = min_pivot < lu.pivot_tolerance

    return NumericalDiagnostics(
        dimension=m,
        min_abs_pivot=min_pivot,
        max_abs_pivot=max_pivot,
        pivot_growth=pivot_growth,
        condition_number=cond_num,
        is_singular=is_singular,
        success=not is_singular,
    )
