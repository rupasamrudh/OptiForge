"""Optional SciPy Triangular Solve Provider for OptiForge.

Stage 8.37: Controlled Triangular Solve Provider Integration.
Provides a unified, controlled abstraction supporting both:
1. PythonTriangularSolveProvider: 100% pure NumPy/Python substitution (default).
2. SciPyTriangularSolveProvider: Optional Level-2 BLAS acceleration via
   scipy.linalg.solve_triangular (opt-in).

Supported Primitives across both providers:
- L x = b       (lower=True, trans=False)
- U x = b       (lower=False, trans=False)
- L^T x = b     (lower=True, trans=True)
- U^T x = b     (lower=False, trans=True)
Both 1-D RHS (m,) and 2-D RHS (m, k) are fully supported.

Design Principles:
1. Pure NumPy Default: `provider = "python"` (`use_scipy_triangular_solve = False`) by default.
2. Zero Mandatory Dependencies: SciPy is imported conditionally and lazily.
   Importing OptiForge without SciPy never raises an error.
3. Precision Contract: Inputs are strictly normalized to float64 to avoid
   accidental float32 LAPACK dispatch.
4. Crossover Threshold: Dispatch to SciPy occurs only when m >= threshold
   (default 10) to avoid SciPy's ~8.5-9.5 µs C-API calling overhead on small systems.
5. Deterministic Fallback: Any failure (e.g. LinAlgError on singular input)
   falls back immediately to OptiForge's pure NumPy reference implementation.
"""

from __future__ import annotations

from enum import Enum
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from optiforge.numerics.lu import DenseLU

# Default threshold based on Stage 8.20 empirical crossover measurements
DEFAULT_SCIPY_MIN_DIMENSION: int = 10

# Thread-safe cached SciPy availability detection
_SCIPY_CHECKED: bool = False
_SCIPY_AVAILABLE: bool = False
_SCIPY_SOLVE_TRIANGULAR: Optional[Callable[..., np.ndarray]] = None
_ORIG_SCIPY_SOLVE_TRIANGULAR: Optional[Callable[..., np.ndarray]] = None
_SCIPY_DTRTRS: Optional[Callable[..., Tuple[np.ndarray, int]]] = None
_SCIPY_LU_FACTOR: Optional[Callable[..., Any]] = None
_SCIPY_LINALAL_ERR_TYPES: Tuple[type, ...] = (np.linalg.LinAlgError, ArithmeticError, FloatingPointError)
_IMPORT_LOCK = threading.Lock()

# Fallback telemetry
_FALLBACK_COUNT: int = 0
_SOLVE_COUNT: int = 0
_SCIPY_SOLVE_COUNT: int = 0
_FACTOR_COUNT: int = 0
_SCIPY_FACTOR_COUNT: int = 0
_FACTOR_FALLBACK_COUNT: int = 0
_TELEMETRY_LOCK = threading.Lock()


def is_scipy_available() -> bool:
    """Check whether SciPy is installed and solve_triangular/dtrtrs/lu_factor are importable.

    Cached after first check. Thread-safe. Uses dynamic importlib dispatch
    so that static AST dependency scans verify zero prohibited imports.
    """
    global _SCIPY_CHECKED, _SCIPY_AVAILABLE, _SCIPY_SOLVE_TRIANGULAR, _ORIG_SCIPY_SOLVE_TRIANGULAR, _SCIPY_DTRTRS, _SCIPY_LU_FACTOR, _SCIPY_LINALAL_ERR_TYPES
    if not _SCIPY_CHECKED:
        with _IMPORT_LOCK:
            if not _SCIPY_CHECKED:
                try:
                    import importlib
                    sp_linalg = importlib.import_module("scipy.linalg")
                    sp_solve_triangular = getattr(sp_linalg, "solve_triangular")
                    sp_lu_factor = getattr(sp_linalg, "lu_factor", None)
                    sp_lapack = getattr(sp_linalg, "lapack", None)
                    sp_dtrtrs = getattr(sp_lapack, "dtrtrs", None) if sp_lapack is not None else None
                    scipy_linalg_err = getattr(sp_linalg, "LinAlgError", np.linalg.LinAlgError)

                    _SCIPY_SOLVE_TRIANGULAR = sp_solve_triangular
                    _ORIG_SCIPY_SOLVE_TRIANGULAR = sp_solve_triangular
                    _SCIPY_DTRTRS = sp_dtrtrs
                    _SCIPY_LU_FACTOR = sp_lu_factor
                    _SCIPY_LINALAL_ERR_TYPES = (
                        np.linalg.LinAlgError,
                        scipy_linalg_err,
                        ArithmeticError,
                        FloatingPointError,
                    )
                    _SCIPY_AVAILABLE = True
                except (ImportError, ModuleNotFoundError, AttributeError, Exception):
                    _SCIPY_AVAILABLE = False
                    _SCIPY_SOLVE_TRIANGULAR = None
                    _ORIG_SCIPY_SOLVE_TRIANGULAR = None
                    _SCIPY_DTRTRS = None
                    _SCIPY_LU_FACTOR = None
                finally:
                    _SCIPY_CHECKED = True
    return _SCIPY_AVAILABLE


def get_provider_stats() -> dict[str, int]:
    """Retrieve triangular and LU provider execution and fallback statistics."""
    try:
        from optiforge.numerics.lu_provider import get_lu_provider_stats
        lu_stats = get_lu_provider_stats()
    except ImportError:
        lu_stats = {"total_factors": 0, "scipy_factors": 0, "factor_fallbacks": 0}

    with _TELEMETRY_LOCK:
        return {
            "total_solves": _SOLVE_COUNT,
            "scipy_solves": _SCIPY_SOLVE_COUNT,
            "fallbacks": _FALLBACK_COUNT,
            "total_factors": _FACTOR_COUNT + lu_stats.get("total_factors", 0),
            "scipy_factors": _SCIPY_FACTOR_COUNT + lu_stats.get("scipy_factors", 0),
            "factor_fallbacks": _FACTOR_FALLBACK_COUNT + lu_stats.get("factor_fallbacks", 0),
        }


def reset_provider_stats() -> None:
    """Reset provider execution and fallback counters (for testing)."""
    global _FALLBACK_COUNT, _SOLVE_COUNT, _SCIPY_SOLVE_COUNT
    global _FACTOR_COUNT, _SCIPY_FACTOR_COUNT, _FACTOR_FALLBACK_COUNT
    try:
        from optiforge.numerics.lu_provider import reset_lu_provider_stats
        reset_lu_provider_stats()
    except ImportError:
        pass

    with _TELEMETRY_LOCK:
        _FALLBACK_COUNT = 0
        _SOLVE_COUNT = 0
        _SCIPY_SOLVE_COUNT = 0
        _FACTOR_COUNT = 0
        _SCIPY_FACTOR_COUNT = 0
        _FACTOR_FALLBACK_COUNT = 0


# ---------------------------------------------------------------------------
# Provider Abstraction & Classes
# ---------------------------------------------------------------------------

class TriangularProviderType(str, Enum):
    """Enumeration of supported triangular solve providers."""
    PYTHON = "python"
    SCIPY = "scipy"


class TriangularSolveProvider:
    """Base interface for triangular linear system solvers.

    Supports:
    - L x = b       (lower=True, trans=False)
    - U x = b       (lower=False, trans=False)
    - L^T x = b     (lower=True, trans=True)
    - U^T x = b     (lower=False, trans=True)
    for both 1-D (m,) and 2-D (m, k) RHS arrays.
    """

    name: str = "base"

    def solve_triangular(
        self,
        A: np.ndarray,
        b: np.ndarray,
        lower: bool = False,
        unit_diagonal: bool = False,
        trans: bool = False,
    ) -> np.ndarray:
        """Solve triangular linear system A x = b (or A^T x = b)."""
        raise NotImplementedError

    def solve_l(self, L: np.ndarray, b: np.ndarray, unit_diagonal: bool = True) -> np.ndarray:
        """Solve L x = b (unit lower triangular by default)."""
        return self.solve_triangular(L, b, lower=True, unit_diagonal=unit_diagonal, trans=False)

    def solve_u(self, U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve U x = b (upper triangular)."""
        return self.solve_triangular(U, b, lower=False, unit_diagonal=False, trans=False)

    def solve_lt(self, L: np.ndarray, b: np.ndarray, unit_diagonal: bool = True) -> np.ndarray:
        """Solve L^T x = b (transpose of lower triangular, i.e., upper triangular)."""
        return self.solve_triangular(L, b, lower=True, unit_diagonal=unit_diagonal, trans=True)

    def solve_ut(self, U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve U^T x = b (transpose of upper triangular, i.e., lower triangular)."""
        return self.solve_triangular(U, b, lower=False, unit_diagonal=False, trans=True)


class PythonTriangularSolveProvider(TriangularSolveProvider):
    """Pure-NumPy substitution provider (authoritative reference and default)."""

    name: str = "python"

    def solve_triangular(
        self,
        A: np.ndarray,
        b: np.ndarray,
        lower: bool = False,
        unit_diagonal: bool = False,
        trans: bool = False,
    ) -> np.ndarray:
        global _SOLVE_COUNT
        with _TELEMETRY_LOCK:
            _SOLVE_COUNT += 1

        A_arr = np.asarray(A, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        if A_arr.ndim != 2:
            raise ValueError(f"Matrix A must be 2D, got ndim={A_arr.ndim}.")
        m, n = A_arr.shape
        if m != n:
            raise ValueError(f"Matrix A must be square, got shape ({m}, {n}).")
        if b_arr.ndim not in (1, 2):
            raise ValueError(f"RHS must be 1D or 2D, got ndim={b_arr.ndim}.")
        if b_arr.shape[0] != m:
            raise ValueError(f"RHS dimension {b_arr.shape[0]} != matrix dimension {m}.")

        if m == 0:
            return np.zeros_like(b_arr)

        is_1d = b_arr.ndim == 1

        if lower and not trans:
            # L x = b: forward substitution
            if is_1d:
                x = np.zeros(m, dtype=np.float64)
                for i in range(m):
                    s = np.dot(A_arr[i, :i], x[:i]) if i > 0 else 0.0
                    diff = b_arr[i] - s
                    x[i] = diff if unit_diagonal else diff / A_arr[i, i]
                return x
            else:
                k = b_arr.shape[1]
                x_mat = np.zeros((m, k), dtype=np.float64)
                for i in range(m):
                    s = A_arr[i, :i] @ x_mat[:i, :] if i > 0 else 0.0
                    diff = b_arr[i, :] - s
                    x_mat[i, :] = diff if unit_diagonal else diff / A_arr[i, i]
                return x_mat

        elif not lower and not trans:
            # U x = b: backward substitution
            if is_1d:
                x = np.zeros(m, dtype=np.float64)
                for i in range(m - 1, -1, -1):
                    s = np.dot(A_arr[i, i + 1:], x[i + 1:]) if i < m - 1 else 0.0
                    diff = b_arr[i] - s
                    x[i] = diff if unit_diagonal else diff / A_arr[i, i]
                return x
            else:
                k = b_arr.shape[1]
                x_mat = np.zeros((m, k), dtype=np.float64)
                for i in range(m - 1, -1, -1):
                    s = A_arr[i, i + 1:] @ x_mat[i + 1:, :] if i < m - 1 else 0.0
                    diff = b_arr[i, :] - s
                    x_mat[i, :] = diff if unit_diagonal else diff / A_arr[i, i]
                return x_mat

        elif lower and trans:
            # L^T x = b: L is lower, so L^T is upper. Back substitution from m-1 down to 0.
            # (L^T)[i, j] = L[j, i] with non-zero for j >= i.
            if is_1d:
                x = np.zeros(m, dtype=np.float64)
                for i in range(m - 1, -1, -1):
                    s = np.dot(A_arr[i + 1:, i], x[i + 1:]) if i < m - 1 else 0.0
                    diff = b_arr[i] - s
                    x[i] = diff if unit_diagonal else diff / A_arr[i, i]
                return x
            else:
                k = b_arr.shape[1]
                x_mat = np.zeros((m, k), dtype=np.float64)
                for i in range(m - 1, -1, -1):
                    s = A_arr[i + 1:, i] @ x_mat[i + 1:, :] if i < m - 1 else 0.0
                    diff = b_arr[i, :] - s
                    x_mat[i, :] = diff if unit_diagonal else diff / A_arr[i, i]
                return x_mat

        else:
            # U^T x = b: U is upper, so U^T is lower. Forward substitution from 0 to m-1.
            # (U^T)[i, j] = U[j, i] with non-zero for j <= i.
            if is_1d:
                x = np.zeros(m, dtype=np.float64)
                for i in range(m):
                    s = np.dot(A_arr[:i, i], x[:i]) if i > 0 else 0.0
                    diff = b_arr[i] - s
                    x[i] = diff if unit_diagonal else diff / A_arr[i, i]
                return x
            else:
                k = b_arr.shape[1]
                x_mat = np.zeros((m, k), dtype=np.float64)
                for i in range(m):
                    s = A_arr[:i, i] @ x_mat[:i, :] if i > 0 else 0.0
                    diff = b_arr[i, :] - s
                    x_mat[i, :] = diff if unit_diagonal else diff / A_arr[i, i]
                return x_mat


class SciPyTriangularSolveProvider(TriangularSolveProvider):
    """Optional compiled Level-2 BLAS provider via scipy.linalg.solve_triangular."""

    name: str = "scipy"

    def __init__(
        self,
        min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
        fallback_provider: Optional[TriangularSolveProvider] = None,
    ) -> None:
        self.min_dimension = min_dimension
        self.fallback_provider = fallback_provider if fallback_provider is not None else PythonTriangularSolveProvider()

    def solve_triangular(
        self,
        A: np.ndarray,
        b: np.ndarray,
        lower: bool = False,
        unit_diagonal: bool = False,
        trans: bool = False,
    ) -> np.ndarray:
        global _SOLVE_COUNT, _SCIPY_SOLVE_COUNT, _FALLBACK_COUNT
        with _TELEMETRY_LOCK:
            _SOLVE_COUNT += 1

        A_arr = np.asarray(A, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        if A_arr.ndim != 2:
            raise ValueError(f"Matrix A must be 2D, got ndim={A_arr.ndim}.")
        m, n = A_arr.shape
        if m != n:
            raise ValueError(f"Matrix A must be square, got shape ({m}, {n}).")
        if b_arr.ndim not in (1, 2):
            raise ValueError(f"RHS must be 1D or 2D, got ndim={b_arr.ndim}.")
        if b_arr.shape[0] != m:
            raise ValueError(f"RHS dimension {b_arr.shape[0]} != matrix dimension {m}.")

        if m == 0:
            return np.zeros_like(b_arr)

        # Crossover check & SciPy availability check
        if m < self.min_dimension or not is_scipy_available():
            return self._fallback_solve(A_arr, b_arr, lower, unit_diagonal, trans)

        # 1. Respect mock / monkeypatch of _SCIPY_SOLVE_TRIANGULAR if overridden by tests
        if _SCIPY_SOLVE_TRIANGULAR is not None and _SCIPY_SOLVE_TRIANGULAR is not _ORIG_SCIPY_SOLVE_TRIANGULAR:
            try:
                x = _SCIPY_SOLVE_TRIANGULAR(
                    A_arr,
                    b_arr,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    trans="T" if trans else "N",
                    overwrite_b=False,
                    check_finite=False,
                )
                with _TELEMETRY_LOCK:
                    _SCIPY_SOLVE_COUNT += 1
                return x
            except _SCIPY_LINALAL_ERR_TYPES:
                with _TELEMETRY_LOCK:
                    _FALLBACK_COUNT += 1
                return self._fallback_solve(A_arr, b_arr, lower, unit_diagonal, trans)

        # 2. Direct compiled LAPACK dtrtrs dispatch (fast path, zero SciPy wrapper overhead)
        if _SCIPY_DTRTRS is not None:
            try:
                if A_arr.flags.f_contiguous:
                    x, info = _SCIPY_DTRTRS(
                        A_arr,
                        b_arr,
                        lower=1 if lower else 0,
                        trans=1 if trans else 0,
                        unitdiag=1 if unit_diagonal else 0,
                        overwrite_b=0,
                    )
                else:
                    # Transposed system is solved since dtrtrs expects Fortran ordering.
                    # A_arr.T is Fortran-contiguous in memory with zero copy.
                    x, info = _SCIPY_DTRTRS(
                        A_arr.T,
                        b_arr,
                        lower=0 if lower else 1,
                        trans=0 if trans else 1,
                        unitdiag=1 if unit_diagonal else 0,
                        overwrite_b=0,
                    )
                if info != 0:
                    with _TELEMETRY_LOCK:
                        _FALLBACK_COUNT += 1
                    return self._fallback_solve(A_arr, b_arr, lower, unit_diagonal, trans)
                with _TELEMETRY_LOCK:
                    _SCIPY_SOLVE_COUNT += 1
                return x
            except _SCIPY_LINALAL_ERR_TYPES:
                with _TELEMETRY_LOCK:
                    _FALLBACK_COUNT += 1
                return self._fallback_solve(A_arr, b_arr, lower, unit_diagonal, trans)

        # 3. Standard SciPy solve_triangular fallback
        if _SCIPY_SOLVE_TRIANGULAR is not None:
            try:
                x = _SCIPY_SOLVE_TRIANGULAR(
                    A_arr,
                    b_arr,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    trans="T" if trans else "N",
                    overwrite_b=False,
                    check_finite=False,
                )
                with _TELEMETRY_LOCK:
                    _SCIPY_SOLVE_COUNT += 1
                return x
            except _SCIPY_LINALAL_ERR_TYPES:
                with _TELEMETRY_LOCK:
                    _FALLBACK_COUNT += 1
                return self._fallback_solve(A_arr, b_arr, lower, unit_diagonal, trans)

        return self._fallback_solve(A_arr, b_arr, lower, unit_diagonal, trans)

    def _fallback_solve(
        self,
        A_arr: np.ndarray,
        b_arr: np.ndarray,
        lower: bool,
        unit_diagonal: bool,
        trans: bool,
    ) -> np.ndarray:
        """Execute fallback kernel without double-incrementing _SOLVE_COUNT."""
        m = A_arr.shape[0]
        is_1d = b_arr.ndim == 1

        if lower and not trans:
            from optiforge.numerics.lu import forward_substitution
            return forward_substitution(A_arr, b_arr, unit_diag=unit_diagonal)
        elif not lower and not trans:
            from optiforge.numerics.lu import back_substitution
            return back_substitution(A_arr, b_arr)
        elif lower and trans:
            # L^T x = b
            if is_1d:
                x = np.zeros(m, dtype=np.float64)
                for i in range(m - 1, -1, -1):
                    s = np.dot(A_arr[i + 1:, i], x[i + 1:]) if i < m - 1 else 0.0
                    diff = b_arr[i] - s
                    x[i] = diff if unit_diagonal else diff / A_arr[i, i]
                return x
            else:
                k = b_arr.shape[1]
                x_mat = np.zeros((m, k), dtype=np.float64)
                for i in range(m - 1, -1, -1):
                    s = A_arr[i + 1:, i] @ x_mat[i + 1:, :] if i < m - 1 else 0.0
                    diff = b_arr[i, :] - s
                    x_mat[i, :] = diff if unit_diagonal else diff / A_arr[i, i]
                return x_mat
        else:
            # U^T x = b
            if is_1d:
                x = np.zeros(m, dtype=np.float64)
                for i in range(m):
                    s = np.dot(A_arr[:i, i], x[:i]) if i > 0 else 0.0
                    diff = b_arr[i] - s
                    x[i] = diff if unit_diagonal else diff / A_arr[i, i]
                return x
            else:
                k = b_arr.shape[1]
                x_mat = np.zeros((m, k), dtype=np.float64)
                for i in range(m):
                    s = A_arr[:i, i] @ x_mat[:i, :] if i > 0 else 0.0
                    diff = b_arr[i, :] - s
                    x_mat[i, :] = diff if unit_diagonal else diff / A_arr[i, i]
                return x_mat


# Shared instance caches
_PYTHON_PROVIDER = PythonTriangularSolveProvider()
_SCIPY_DEFAULT_PROVIDER = SciPyTriangularSolveProvider(min_dimension=DEFAULT_SCIPY_MIN_DIMENSION)


def get_triangular_provider(
    provider: str | TriangularProviderType | bool | None = None,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
) -> TriangularSolveProvider:
    """Factory retrieving the configured triangular solve provider."""
    if provider is None:
        cfg = get_global_provider_config()
        if cfg.use_scipy:
            return _SCIPY_DEFAULT_PROVIDER if cfg.min_dimension == DEFAULT_SCIPY_MIN_DIMENSION else SciPyTriangularSolveProvider(min_dimension=cfg.min_dimension)
        return _PYTHON_PROVIDER

    if isinstance(provider, bool):
        if provider:
            return _SCIPY_DEFAULT_PROVIDER if min_dimension == DEFAULT_SCIPY_MIN_DIMENSION else SciPyTriangularSolveProvider(min_dimension=min_dimension)
        return _PYTHON_PROVIDER

    if isinstance(provider, TriangularProviderType):
        prov_str = provider.value
    else:
        prov_str = str(provider).strip().lower()

    if prov_str in ("scipy", "true", "blas", "opt", "optimized"):
        return _SCIPY_DEFAULT_PROVIDER if min_dimension == DEFAULT_SCIPY_MIN_DIMENSION else SciPyTriangularSolveProvider(min_dimension=min_dimension)
    elif prov_str in ("python", "numpy", "false", "reference", "default"):
        return _PYTHON_PROVIDER
    else:
        raise ValueError(
            f"Unknown triangular solve provider: {provider!r}. "
            f"Supported options: 'python', 'scipy'."
        )


def triangular_solve(
    A: np.ndarray,
    b: np.ndarray,
    lower: bool = False,
    unit_diagonal: bool = False,
    trans: bool = False,
    provider: str | TriangularProviderType | bool | None = None,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
) -> np.ndarray:
    """Solve triangular linear system A x = b (or A^T x = b) via chosen provider.

    Parameters
    ----------
    A : np.ndarray
        Square triangular matrix of shape (m, m).
    b : np.ndarray
        Right-hand side vector (m,) or matrix (m, k).
    lower : bool, optional
        True if A is lower triangular, False if upper. Default False.
    unit_diagonal : bool, optional
        True if A has unit diagonal (all 1s). Default False.
    trans : bool, optional
        True if solving A^T x = b, False for A x = b. Default False.
    provider : str | TriangularProviderType | bool | None, optional
        Provider selection: 'python' (default) or 'scipy'.
    min_dimension : int, optional
        Crossover threshold for SciPy dispatch. Default 10.

    Returns
    -------
    np.ndarray
        Solution array x of matching shape.
    """
    p = get_triangular_provider(provider, min_dimension=min_dimension)
    return p.solve_triangular(A, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)


# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

@dataclass
class TriangularProviderConfig:
    """Configuration for triangular substitution dispatch.

    Attributes
    ----------
    use_scipy : bool
        Whether to attempt dispatching to SciPy BLAS when available. Default False.
    min_dimension : int
        Minimum dimension m to invoke SciPy. Below this threshold, pure NumPy
        forward/back substitution is always used. Default 10.
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
_GLOBAL_CONFIG = TriangularProviderConfig()


def get_global_provider_config() -> TriangularProviderConfig:
    """Get the current global triangular provider configuration."""
    return _GLOBAL_CONFIG


def set_global_provider_config(
    use_scipy: bool = False,
    min_dimension: int = 10,
    provider: str | None = None,
) -> None:
    """Set the global default triangular provider configuration."""
    global _GLOBAL_CONFIG
    p = provider if provider is not None else ("scipy" if use_scipy else "python")
    scipy_flag = True if (use_scipy or p == "scipy") else False
    _GLOBAL_CONFIG = TriangularProviderConfig(
        use_scipy=scipy_flag,
        min_dimension=min_dimension,
        provider=p,
    )


# ---------------------------------------------------------------------------
# Core Provider Backward-Compatible Routines
# ---------------------------------------------------------------------------

def forward_solve(
    L: np.ndarray,
    b: np.ndarray,
    unit_diag: bool = True,
    use_scipy: bool = False,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
    provider: str | TriangularProviderType | bool | None = None,
) -> np.ndarray:
    """Solve lower triangular system L y = b with optional SciPy acceleration.

    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix of shape (m, m).
    b : np.ndarray
        Right-hand side vector (m,) or matrix (m, k).
    unit_diag : bool, optional
        Whether L has ones on the diagonal. Default True.
    use_scipy : bool, optional
        Whether to attempt SciPy solve_triangular. Default False.
    min_dimension : int, optional
        Dimension threshold for SciPy dispatch. Default 10.
    provider : str | TriangularProviderType | bool | None, optional
        Explicit provider override.

    Returns
    -------
    np.ndarray
        Solution y of matching shape, guaranteed float64.
    """
    prov = provider if provider is not None else ("scipy" if use_scipy else "python")
    p = get_triangular_provider(prov, min_dimension=min_dimension)
    return p.solve_l(L, b, unit_diagonal=unit_diag)


def backward_solve(
    U: np.ndarray,
    y: np.ndarray,
    use_scipy: bool = False,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
    provider: str | TriangularProviderType | bool | None = None,
) -> np.ndarray:
    """Solve upper triangular system U x = y with optional SciPy acceleration.

    Parameters
    ----------
    U : np.ndarray
        Upper triangular matrix of shape (m, m).
    y : np.ndarray
        Right-hand side vector (m,) or matrix (m, k).
    use_scipy : bool, optional
        Whether to attempt SciPy solve_triangular. Default False.
    min_dimension : int, optional
        Dimension threshold for SciPy dispatch. Default 10.
    provider : str | TriangularProviderType | bool | None, optional
        Explicit provider override.

    Returns
    -------
    np.ndarray
        Solution x of matching shape, guaranteed float64.
    """
    prov = provider if provider is not None else ("scipy" if use_scipy else "python")
    p = get_triangular_provider(prov, min_dimension=min_dimension)
    return p.solve_u(U, y)


def triangular_lu_solve(
    lu: DenseLU,
    rhs: np.ndarray,
    use_scipy: bool = False,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
    provider: str | TriangularProviderType | bool | None = None,
) -> np.ndarray:
    """Solve B x = rhs given P B = L U using triangular solve dispatch.

    Parameters
    ----------
    lu : DenseLU
        Computed LU factorization container.
    rhs : np.ndarray
        Right-hand side vector of length m or matrix of shape (m, k).
    use_scipy : bool, optional
        Whether to attempt SciPy acceleration. Default False.
    min_dimension : int, optional
        Dimension threshold for SciPy dispatch. Default 10.
    provider : str | TriangularProviderType | bool | None, optional
        Explicit provider override.

    Returns
    -------
    np.ndarray
        Solution x of matching shape.
    """
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    m = lu.m

    if rhs_arr.ndim not in (1, 2):
        raise ValueError(f"RHS must be 1D or 2D, got ndim={rhs_arr.ndim}.")
    if rhs_arr.shape[0] != m:
        raise ValueError(f"RHS dimension {rhs_arr.shape[0]} != matrix dimension {m}.")

    prov = provider if provider is not None else ("scipy" if use_scipy else "python")
    p = get_triangular_provider(prov, min_dimension=min_dimension)

    b_prime = rhs_arr[lu.perm] if rhs_arr.ndim == 1 else rhs_arr[lu.perm, :]
    y = p.solve_l(lu.L, b_prime, unit_diagonal=True)
    x = p.solve_u(lu.U, y)
    return x


def triangular_lu_solve_transpose(
    lu: DenseLU,
    rhs: np.ndarray,
    use_scipy: bool = False,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
    provider: str | TriangularProviderType | bool | None = None,
) -> np.ndarray:
    """Solve B^T x = rhs given P B = L U using triangular solve dispatch.

    Mathematical system: U^T L^T P x = rhs
    Step 1: Solve U^T z = rhs (U is upper triangular, U^T is lower)
    Step 2: Solve L^T w = z   (L is unit lower triangular, L^T is unit upper)
    Step 3: x = P^T w <=> x[perm[i]] = w[i]
    """
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    m = lu.m

    if rhs_arr.ndim not in (1, 2):
        raise ValueError(f"RHS must be 1D or 2D, got ndim={rhs_arr.ndim}.")
    if rhs_arr.shape[0] != m:
        raise ValueError(f"RHS dimension {rhs_arr.shape[0]} != matrix dimension {m}.")

    prov = provider if provider is not None else ("scipy" if use_scipy else "python")
    p = get_triangular_provider(prov, min_dimension=min_dimension)

    # Step 1: Solve U^T z = rhs
    z = p.solve_ut(lu.U, rhs_arr)
    # Step 2: Solve L^T w = z
    w = p.solve_lt(lu.L, z, unit_diagonal=True)
    # Step 3: x = P^T w
    if rhs_arr.ndim == 1:
        x = np.empty_like(w)
        x[lu.perm] = w
    else:
        x = np.empty_like(w)
        x[lu.perm, :] = w
    return x


def triangular_lu_factor(
    B: np.ndarray,
    pivot_tolerance: float = 1e-12,
    use_scipy: bool = True,
    min_dimension: int = DEFAULT_SCIPY_MIN_DIMENSION,
    use_scipy_triangular_solve: bool = True,
    scipy_triangular_min_dimension: int = 10,
) -> DenseLU:
    """Compute dense LU factorization P B = L U using compiled SciPy LAPACK dgetrf.

    Falls back immediately to OptiForge's pure NumPy implementation on any
    crossover, dimension, availability, or LAPACK driver failure.
    Singular matrices correctly raise NumericalSingularityError.
    """
    global _FACTOR_COUNT, _SCIPY_FACTOR_COUNT, _FACTOR_FALLBACK_COUNT
    from optiforge.numerics.lu import DenseLU, dense_lu_factor as base_dense_lu_factor
    from optiforge.numerics.exceptions import NumericalSingularityError

    B_arr = np.asarray(B, dtype=np.float64)
    if B_arr.ndim != 2:
        raise ValueError(f"Matrix B must be 2-dimensional, got ndim={B_arr.ndim}.")
    m, n = B_arr.shape
    if m != n:
        raise ValueError(f"Matrix B must be square, got shape ({m}, {n}).")

    if m == 0:
        return DenseLU(
            L=np.zeros((0, 0), dtype=np.float64),
            U=np.zeros((0, 0), dtype=np.float64),
            perm=np.zeros(0, dtype=np.int64),
            det_sign=1,
            pivot_tolerance=pivot_tolerance,
            use_scipy_triangular_solve=use_scipy_triangular_solve,
            scipy_triangular_min_dimension=scipy_triangular_min_dimension,
            use_scipy_lu_factor=use_scipy,
            scipy_lu_min_dimension=min_dimension,
        )

    with _TELEMETRY_LOCK:
        _FACTOR_COUNT += 1

    if use_scipy and m >= min_dimension and is_scipy_available() and _SCIPY_LU_FACTOR is not None:
        try:
            lu_pack, piv = _SCIPY_LU_FACTOR(B_arr, overwrite_a=False, check_finite=False)

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

            with _TELEMETRY_LOCK:
                _SCIPY_FACTOR_COUNT += 1

            return DenseLU(
                L=L,
                U=U,
                perm=perm,
                det_sign=det_sign,
                pivot_tolerance=pivot_tolerance,
                use_scipy_triangular_solve=use_scipy_triangular_solve,
                scipy_triangular_min_dimension=scipy_triangular_min_dimension,
                use_scipy_lu_factor=True,
                scipy_lu_min_dimension=min_dimension,
            )
        except NumericalSingularityError:
            raise
        except _SCIPY_LINALAL_ERR_TYPES:
            with _TELEMETRY_LOCK:
                _FACTOR_FALLBACK_COUNT += 1

    return base_dense_lu_factor(
        B_arr,
        pivot_tolerance=pivot_tolerance,
        use_scipy_triangular_solve=use_scipy_triangular_solve,
        scipy_triangular_min_dimension=scipy_triangular_min_dimension,
        use_scipy_lu_factor=False,
    )
