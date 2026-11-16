"""Pure linear algebra utilities for the domain layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import norm

from neuralls.shared.types import MatrixNormType


def calculate_spectral_norm(A: np.ndarray) -> float:
    """Calculate spectral norm (largest singular value) of matrix A.

    The spectral norm is defined as the largest singular value of A,
    which equals the square root of the largest eigenvalue of A^T A.
    This is computed using ||A||_2.

    Args:
        A: System matrix

    Returns:
        Spectral norm (largest singular value)
    """
    return float(np.linalg.norm(A, ord=2))


def calculate_frobenius_norm(A: np.ndarray) -> float:
    """Calculate Frobenius norm of matrix A.

    The Frobenius norm is defined as the square root of the sum of
    squared entries: ||A||_F = sqrt(sum(A_ij^2)).

    Args:
        A: System matrix

    Returns:
        Frobenius norm
    """
    return float(np.linalg.norm(A, ord="fro"))


def calculate_nuclear_norm(A: np.ndarray) -> float:
    """Calculate nuclear norm (trace norm) of matrix A.

    The nuclear norm is the sum of singular values of A.
    It's the dual norm of the spectral norm.

    Args:
        A: System matrix

    Returns:
        Nuclear norm (sum of singular values)
    """
    return float(np.linalg.norm(A, ord="nuc"))


def calculate_one_norm(A: np.ndarray) -> float:
    """Calculate 1-norm of matrix A.

    The 1-norm is the maximum absolute column sum:
    ||A||_1 = max_j sum_i |A_ij|

    Args:
        A: System matrix

    Returns:
        1-norm (max column sum)
    """
    return float(np.linalg.norm(A, ord=1))


def calculate_inf_norm(A: np.ndarray) -> float:
    """Calculate infinity norm of matrix A.

    The infinity norm is the maximum absolute row sum:
    ||A||_inf = max_i sum_j |A_ij|

    Args:
        A: System matrix

    Returns:
        Infinity norm (max row sum)
    """
    return float(np.linalg.norm(A, ord=np.inf))


# Norm strategy mapping - uses StrEnum as single source of truth
_NORM_STRATEGIES: dict[str, Callable[[np.ndarray], float]] = {
    MatrixNormType.SPECTRAL: calculate_spectral_norm,
    MatrixNormType.FROBENIUS: calculate_frobenius_norm,
    MatrixNormType.NUCLEAR: calculate_nuclear_norm,
    MatrixNormType.ONE: calculate_one_norm,
    MatrixNormType.INF: calculate_inf_norm,
}


def calculate_matrix_norm(
    A: np.ndarray,
    norm_type: str = "spectral",
) -> float:
    """Calculate matrix norm using specified norm type.

    Extensible dispatch function using StrEnum for type safety.
    Users pass strings - StrEnum provides validation internally.

    To add a new norm type:
    1. Add member to MatrixNormType enum above
    2. Add calculate_<name>_norm() function above
    3. Add entry to _NORM_STRATEGIES mapping

    Args:
        A: System matrix
        norm_type: Type of norm to compute (default: "spectral").
            Supported values:
            - "spectral": Spectral norm (2-norm, largest singular value)
            - "frobenius": Frobenius norm (sqrt of sum of squared entries)
            - "nuclear": Nuclear norm (sum of singular values)
            - "one": 1-norm (max column sum)
            - "inf": Infinity norm (max row sum)

    Returns:
        Matrix norm value

    Raises:
        ValueError: If norm_type is not recognized

    Examples:
        >>> A = np.array([[3, 0], [0, 4]])
        >>> calculate_matrix_norm(A, "spectral")  # Max singular value
        4.0
        >>> calculate_matrix_norm(A, "frobenius")  # sqrt(3^2 + 4^2)
        5.0
        >>> calculate_matrix_norm(A, "one")  # max(3, 4)
        4.0
    """
    if norm_type not in _NORM_STRATEGIES:
        supported = ", ".join(norm.value for norm in MatrixNormType)
        raise ValueError(f"Unknown matrix norm type: {norm_type!r}. Supported types: {supported}")

    norm_fn = _NORM_STRATEGIES[norm_type]
    return norm_fn(A)


def compute_dim_scale(dimension: int) -> float:
    """Calculate dimension-based scaling factor, sqrt(dimension).

    ``GershgorinScaling(dimension_factor=SQRT_N)`` multiplies the Gershgorin bound by it,
    so scaled matrices of different sizes have comparable magnitudes.

    Args:
        dimension: Matrix dimension.

    Returns:
        sqrt(dimension).
    """
    return float(np.sqrt(dimension))


def vector_stats(v: np.ndarray) -> dict[str, float]:
    """Compute statistics for a vector.

    Args:
        v: Input vector

    Returns:
        Dictionary with vector statistics
    """
    return {
        "norm": float(norm(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def compute_rhs_norm(rhs: np.ndarray) -> float:
    """Compute L2 norm of RHS vector.

    Args:
        rhs: RHS vector

    Returns:
        L2 norm of rhs
    """
    return float(np.linalg.norm(rhs, ord=2))


def compute_rhs_norms(rhs_samples: np.ndarray) -> np.ndarray:
    """Compute L2 norms for batch of RHS vectors.

    Args:
        rhs_samples: Array of RHS vectors, shape (N, n)

    Returns:
        Array of L2 norms, shape (N,)
    """
    return np.array([compute_rhs_norm(rhs) for rhs in rhs_samples])
