"""Controlled Dense LU Factorization Provider for OptiForge.

Stage 8.39: Controlled LU Factorization Provider Integration.
Provides a unified, controlled abstraction supporting both:
1. PythonLUFactorizationProvider: 100% pure NumPy/Python Gaussian elimination
   with partial pivoting (authoritative reference and default).
2. SciPyLUFactorizationProvider: Optional compiled Level-3 LAPACK dgetrf
   acceleration via scipy.linalg.lu_factor (opt-in).

Design Principles:
1. Pure NumPy Default: provider = "python" (use_scipy_lu_factor = False) by default.
2. Zero Mandatory Dependencies: SciPy is imported conditionally and lazily.
   Importing OptiForge without SciPy never raises an error.
3. Decoupled Architecture: LU factorization provider is completely independent
   of the triangular solve provider.
4. Precision Contract: Inputs are strictly normalized to float64, and output
   DenseLU contract (L unit lower triangular, U upper triangular, perm mapping,
   det_sign) is 100% preserved.
5. Crossover Threshold: Dispatch to SciPy occurs only when m >= threshold
   (default 10) to avoid SciPy C-API calling overhead on small systems.
6. Deterministic Singularity & Fallback: Singular matrices trigger
   NumericalSingularityError matching OptiForge's exact contract.
   Any driver/LAPACK failure safely falls back to OptiForge's reference implementation.
"""

from __future__ import annotations

from enum import Enum
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple
import warnings

import numpy as np

from optiforge.numerics.exceptions import NumericalSingularityError
from optiforge.numerics import triangular_provider as tp
from optiforge.numerics.triangular_provider import (
    DEFAULT_SCIPY_MIN_DIMENSION,
    is_scipy_available,
)

if TYPE_CHECKING:
    from optiforge.numerics.lu import DenseLU

# Telemetry for LU factorization
_LU_FACTOR_COUNT: int = 0
_LU_SCIPY_FACTOR_COUNT: int = 0
_LU_FACTOR_FALLBACK_COUNT: int = 0
_LU_TELEMETRY_LOCK = threading.Lock()


def get_lu_provider_stats() -> dict[str, int]:
    """Retrieve LU provider execution and fallback statistics."""
    with _LU_TELEMETRY_LOCK:
        return {
            "total_factors": _LU_FACTOR_COUNT,
            "scipy_factors": _LU_SCIPY_FACTOR_COUNT,
            "factor_fallbacks": _LU_FACTOR_FALLBACK_COUNT,
        }


def reset_lu_provider_stats() -> None:
    """Reset LU provider execution and fallback counters (for testing)."""
    global _LU_FACTOR_COUNT, _LU_SCIPY_FACTOR_COUNT, _LU_FACTOR_FALLBACK_COUNT
    with _LU_TELEMETRY_LOCK:
        _LU_FACTOR_COUNT = 0
        _LU_SCIPY_FACTOR_COUNT = 0
        _LU_FACTOR_FALLBACK_COUNT = 0


# ---------------------------------------------------------------------------
# Provider Abstraction & Classes
# ---------------------------------------------------------------------------

class LUProviderType(str, Enum):
    """Enumeration of supported LU factorization providers."""
    PYTHON = "python"
    SCIPY = "scipy"


class LUFactorizationProvider:
    """Base interface for dense LU factorization providers."""

    name: str = "base"

    def factorize(
        self,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        **kwargs: Any,
    ) -> DenseLU:
        """Compute dense LU factorization P B = L U."""
        raise NotImplementedError


class PythonLUFactorizationProvider(LUFactorizationProvider):
    """Pure-NumPy vectorized Gaussian elimination provider (authoritative reference and default)."""

    name: str = "python"

    def factorize(
        self,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        **kwargs: Any,
    ) -> DenseLU:
        global _LU_FACTOR_COUNT
        with _LU_TELEMETRY_LOCK:
            _LU_FACTOR_COUNT += 1

        from optiforge.numerics.lu import DenseLU

        B_arr = np.asarray(B, dtype=np.float64)
        if B_arr.ndim != 2:
            raise ValueError(f"Matrix B must be 2-dimensional, got ndim={B_arr.ndim}.")
        m, n = B_arr.shape
        if m != n:
            raise ValueError(f"Matrix B must be square, got shape ({m}, {n}).")

        eff_tri_prov = "scipy" if (use_scipy_triangular_solve or triangular_solve_provider == "scipy") else "python"
        eff_use_scipy_tri = (eff_tri_prov == "scipy")

        if m == 0:
            return DenseLU(
                L=np.zeros((0, 0), dtype=np.float64),
                U=np.zeros((0, 0), dtype=np.float64),
                perm=np.zeros(0, dtype=np.int64),
                det_sign=1,
                pivot_tolerance=pivot_tolerance,
                use_scipy_triangular_solve=eff_use_scipy_tri,
                scipy_triangular_min_dimension=scipy_triangular_min_dimension,
                use_scipy_lu_factor=False,
                scipy_lu_min_dimension=10,
                triangular_solve_provider=eff_tri_prov,
                lu_factor_provider="python",
            )

        A = np.array(B_arr, copy=True, dtype=np.float64, order="C")
        perm = np.arange(m, dtype=np.int64)
        det_sign = 1

        for k in range(m):
            col_abs = np.abs(A[k:, k])
            p_rel = int(col_abs.argmax())
            p = k + p_rel
            pivot_val = A[p, k]
            pivot_mag = abs(pivot_val)

            if pivot_mag < pivot_tolerance:
                raise NumericalSingularityError(
                    f"Matrix is singular or numerically singular at column {k}: "
                    f"pivot magnitude {pivot_mag:.3e} below tolerance {pivot_tolerance:.3e}.",
                    pivot=float(pivot_val),
                    step=k,
                )

            if p != k:
                row_tmp = A[k, :].copy()
                A[k, :] = A[p, :]
                A[p, :] = row_tmp
                perm[k], perm[p] = perm[p], perm[k]
                det_sign = -det_sign

            if k < m - 1:
                pivot = A[k, k]
                factors = A[k + 1:, k] / pivot
                A[k + 1:, k] = factors
                A[k + 1:, k + 1:] -= factors[:, None] * A[k, k + 1:]

        L = np.tril(A, -1) + np.eye(m, dtype=np.float64)
        U = np.triu(A)

        return DenseLU(
            L=L,
            U=U,
            perm=perm,
            det_sign=det_sign,
            pivot_tolerance=pivot_tolerance,
            use_scipy_triangular_solve=eff_use_scipy_tri,
            scipy_triangular_min_dimension=scipy_triangular_min_dimension,
            use_scipy_lu_factor=False,
            scipy_lu_min_dimension=10,
            triangular_solve_provider=eff_tri_prov,
            lu_factor_provider="python",
        )


class SciPyLUFactorizationProvider(LUFactorizationProvider):
    """Optional compiled Level-3 LAPACK dgetrf provider via scipy.linalg.lu_factor."""

    name: str = "scipy"

    def __init__(
        self,
        min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
        fallback_provider: Optional[LUFactorizationProvider] = None,
    ) -> None:
        self.min_dimension = min_dimension
        self.fallback_provider = fallback_provider if fallback_provider is not None else PythonLUFactorizationProvider()

    def factorize(
        self,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        **kwargs: Any,
    ) -> DenseLU:
        global _LU_FACTOR_COUNT, _LU_SCIPY_FACTOR_COUNT, _LU_FACTOR_FALLBACK_COUNT
        with _LU_TELEMETRY_LOCK:
            _LU_FACTOR_COUNT += 1

        from optiforge.numerics.lu import DenseLU

        B_arr = np.asarray(B, dtype=np.float64)
        if B_arr.ndim != 2:
            raise ValueError(f"Matrix B must be 2-dimensional, got ndim={B_arr.ndim}.")
        m, n = B_arr.shape
        if m != n:
            raise ValueError(f"Matrix B must be square, got shape ({m}, {n}).")

        eff_tri_prov = "scipy" if (use_scipy_triangular_solve or triangular_solve_provider == "scipy") else "python"
        eff_use_scipy_tri = (eff_tri_prov == "scipy")

        if m == 0:
            return DenseLU(
                L=np.zeros((0, 0), dtype=np.float64),
                U=np.zeros((0, 0), dtype=np.float64),
                perm=np.zeros(0, dtype=np.int64),
                det_sign=1,
                pivot_tolerance=pivot_tolerance,
                use_scipy_triangular_solve=eff_use_scipy_tri,
                scipy_triangular_min_dimension=scipy_triangular_min_dimension,
                use_scipy_lu_factor=True,
                scipy_lu_min_dimension=self.min_dimension,
                triangular_solve_provider=eff_tri_prov,
                lu_factor_provider="scipy",
            )

        # Crossover threshold & SciPy availability check
        if m < self.min_dimension or not is_scipy_available() or tp._SCIPY_LU_FACTOR is None:
            return self._fallback_factorize(
                B_arr,
                pivot_tolerance=pivot_tolerance,
                use_scipy_triangular_solve=eff_use_scipy_tri,
                scipy_triangular_min_dimension=scipy_triangular_min_dimension,
                triangular_solve_provider=eff_tri_prov,
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lu_pack, piv = tp._SCIPY_LU_FACTOR(B_arr, overwrite_a=False, check_finite=False)

            # Check singularity along U diagonal
            for k in range(m):
                val = lu_pack[k, k]
                mag = abs(val)
                if mag < pivot_tolerance:
                    raise NumericalSingularityError(
                        f"Matrix is singular or numerically singular at column {k}: "
                        f"pivot magnitude {mag:.3e} below tolerance {pivot_tolerance:.3e}.",
                        pivot=float(val),
                        step=k,
                    )

            L = np.tril(lu_pack, -1) + np.eye(m, dtype=np.float64)
            U = np.triu(lu_pack)

            perm = np.arange(m, dtype=np.int64)
            det_sign = 1
            for i in range(m):
                p = int(piv[i])
                if p != i:
                    perm[i], perm[p] = perm[p], perm[i]
                    det_sign = -det_sign

            with _LU_TELEMETRY_LOCK:
                _LU_SCIPY_FACTOR_COUNT += 1

            return DenseLU(
                L=L,
                U=U,
                perm=perm,
                det_sign=det_sign,
                pivot_tolerance=pivot_tolerance,
                use_scipy_triangular_solve=eff_use_scipy_tri,
                scipy_triangular_min_dimension=scipy_triangular_min_dimension,
                use_scipy_lu_factor=True,
                scipy_lu_min_dimension=self.min_dimension,
                triangular_solve_provider=eff_tri_prov,
                lu_factor_provider="scipy",
            )
        except NumericalSingularityError:
            raise
        except tp._SCIPY_LINALAL_ERR_TYPES:
            with _LU_TELEMETRY_LOCK:
                _LU_FACTOR_FALLBACK_COUNT += 1
            return self._fallback_factorize(
                B_arr,
                pivot_tolerance=pivot_tolerance,
                use_scipy_triangular_solve=eff_use_scipy_tri,
                scipy_triangular_min_dimension=scipy_triangular_min_dimension,
                triangular_solve_provider=eff_tri_prov,
            )

    def _fallback_factorize(
        self,
        B_arr: np.ndarray,
        pivot_tolerance: float,
        use_scipy_triangular_solve: bool,
        scipy_triangular_min_dimension: int,
        triangular_solve_provider: str,
    ) -> DenseLU:
        """Execute fallback kernel without double-incrementing _LU_FACTOR_COUNT."""
        # Use PythonLUFactorizationProvider directly without re-incrementing count
        from optiforge.numerics.lu import DenseLU

        m = B_arr.shape[0]
        A = np.array(B_arr, copy=True, dtype=np.float64, order="C")
        perm = np.arange(m, dtype=np.int64)
        det_sign = 1

        for k in range(m):
            col_abs = np.abs(A[k:, k])
            p_rel = int(col_abs.argmax())
            p = k + p_rel
            pivot_val = A[p, k]
            pivot_mag = abs(pivot_val)

            if pivot_mag < pivot_tolerance:
                raise NumericalSingularityError(
                    f"Matrix is singular or numerically singular at column {k}: "
                    f"pivot magnitude {pivot_mag:.3e} below tolerance {pivot_tolerance:.3e}.",
                    pivot=float(pivot_val),
                    step=k,
                )

            if p != k:
                row_tmp = A[k, :].copy()
                A[k, :] = A[p, :]
                A[p, :] = row_tmp
                perm[k], perm[p] = perm[p], perm[k]
                det_sign = -det_sign

            if k < m - 1:
                pivot = A[k, k]
                factors = A[k + 1:, k] / pivot
                A[k + 1:, k] = factors
                A[k + 1:, k + 1:] -= factors[:, None] * A[k, k + 1:]

        L = np.tril(A, -1) + np.eye(m, dtype=np.float64)
        U = np.triu(A)

        return DenseLU(
            L=L,
            U=U,
            perm=perm,
            det_sign=det_sign,
            pivot_tolerance=pivot_tolerance,
            use_scipy_triangular_solve=use_scipy_triangular_solve,
            scipy_triangular_min_dimension=scipy_triangular_min_dimension,
            use_scipy_lu_factor=False,
            scipy_lu_min_dimension=self.min_dimension,
            triangular_solve_provider=triangular_solve_provider,
            lu_factor_provider="python",
        )


# Shared instance caches
_PYTHON_LU_PROVIDER = PythonLUFactorizationProvider()


# ---------------------------------------------------------------------------
# Global Configuration & Factory
# ---------------------------------------------------------------------------

@dataclass
class LUProviderConfig:
    """Configuration for LU factorization dispatch.

    Attributes
    ----------
    use_scipy : bool
        Whether to attempt dispatching to SciPy LAPACK dgetrf when available. Default False.
    min_dimension : int
        Minimum dimension m to invoke SciPy. Default 10.
    provider : str
        Canonical provider name ('python' or 'scipy').
    """

    use_scipy: bool = False
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION
    provider: str = "python"

    def __post_init__(self) -> None:
        if self.use_scipy:
            self.provider = "scipy"
        elif self.provider == "scipy":
            self.use_scipy = True


# Global default configuration (Python reference default)
_GLOBAL_LU_CONFIG = LUProviderConfig()


def get_global_lu_provider_config() -> LUProviderConfig:
    """Get the current global LU factorization provider configuration."""
    return _GLOBAL_LU_CONFIG


def set_global_lu_provider_config(
    use_scipy: bool = False,
    min_dimension: int = 10,
    provider: str | None = None,
) -> None:
    """Set the global default LU factorization provider configuration."""
    global _GLOBAL_LU_CONFIG
    p = provider if provider is not None else ("scipy" if use_scipy else "python")
    _GLOBAL_LU_CONFIG = LUProviderConfig(
        use_scipy=use_scipy or (p == "scipy"),
        min_dimension=min_dimension,
        provider=p,
    )


def get_lu_provider(
    provider: str | LUProviderType | bool | None = None,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
) -> LUFactorizationProvider:
    """Factory retrieving the configured LU factorization provider."""
    if provider is None:
        cfg = get_global_lu_provider_config()
        if cfg.use_scipy:
            return SciPyLUFactorizationProvider(min_dimension=cfg.min_dimension)
        return _PYTHON_LU_PROVIDER

    if isinstance(provider, bool):
        if provider:
            return SciPyLUFactorizationProvider(min_dimension=min_dimension)
        return _PYTHON_LU_PROVIDER

    if isinstance(provider, LUProviderType):
        prov_str = provider.value
    else:
        prov_str = str(provider).strip().lower()

    if prov_str in ("scipy", "true", "lapack", "dgetrf", "opt", "optimized"):
        return SciPyLUFactorizationProvider(min_dimension=min_dimension)
    elif prov_str in ("python", "numpy", "false", "reference", "default"):
        return _PYTHON_LU_PROVIDER
    else:
        raise ValueError(
            f"Unknown LU factorization provider: {provider!r}. "
            f"Supported options: 'python', 'scipy'."
        )
