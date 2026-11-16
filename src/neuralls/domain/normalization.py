"""System scaling arithmetic and sparse-aware norms for the domain layer.

A system is scaled by one positive scalar per matrix, ``alpha``: ``A' = A / alpha`` and
``b' = b / alpha``. For ``GershgorinScaling`` ``alpha = max(||A||_1, ||A||_inf) * f(n)``
with ``f(n)`` equal to 1 or ``sqrt(n)``. ``max(||A||_1, ||A||_inf)`` is a Gershgorin bound
on the spectral radius, not the spectral radius itself. A scalar scaling leaves the
solution, the CG iterates and weighted-Jacobi sweeps with ``omega = 1 / rho(D^-1 A)``
unchanged; it fixes only the magnitude of ``A``, ``b`` and the residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse.linalg import svds

from neuralls.domain.linalg import calculate_matrix_norm, compute_dim_scale
from neuralls.shared.system_scaling import (
    DimensionFactor,
    GershgorinScaling,
    NoScaling,
    SystemScaling,
)
from neuralls.shared.types import MatrixNormType, SystemMatrix

# Sparse norm orders passed to scipy.sparse.linalg.norm. Nuclear is absent
# because scipy has no sparse implementation of it.
_SPARSE_NORM_ORDER: Final[dict[MatrixNormType, Literal["fro"] | float]] = {
    MatrixNormType.FROBENIUS: "fro",
    MatrixNormType.ONE: 1,
    MatrixNormType.INF: np.inf,
}

# =============================================================================
# Frozen Dataclasses: Trace Samples
# =============================================================================


@dataclass(frozen=True, slots=True)
class ResidualTraceSamples:
    """Residual → solution trace pairs from CG iterations.

    Captures residuals r_k and corresponding solutions x_k at iteration k.
    Training mapping: N(r_k) → x_k
    """

    residuals: np.ndarray
    solutions: np.ndarray
    sample_indices: np.ndarray
    iteration_indices: np.ndarray
    search_directions: np.ndarray | None = None
    search_direction_products: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class ErrorTraceSamples:
    """Error correction traces: r_k → (x* - x_k).

    Training mapping: N(r_k) → (x* - x_k) where r_k is the network input.
    solutions_current (x_k) is stored for error computation but NOT as input.
    """

    residuals: np.ndarray
    solutions_current: np.ndarray  # x_k (for error computation only)
    errors: np.ndarray  # x* - x_k (training targets)
    true_solutions: np.ndarray  # x* per sample
    sample_indices: np.ndarray
    iteration_indices: np.ndarray


# =============================================================================
# Pure Functions: Sparse-aware scaling and norms
# =============================================================================


def scale_csr_values(
    matrix: csr_array,
    scale: float | np.ndarray,
    axis: Literal["row", "col"] | None,
) -> csr_array:
    """Divide the stored values of a CSR matrix by a scale, keeping its pattern.

    Only ``data`` is recomputed; ``indices`` and ``indptr`` are shared with the
    input, so the sparsity pattern (including explicit zeros) is unchanged and
    the matrix is never densified.

    Args:
        matrix: Sparse system matrix.
        scale: A scalar when ``axis`` is None, otherwise one factor per row
            (``axis="row"``) or per column (``axis="col"``).
        axis: Which dimension ``scale`` indexes, or None for a global scalar.

    Returns:
        New ``csr_array`` with ``data / scale`` broadcast along ``axis``.

    Raises:
        ValueError: If ``scale`` is an array with the wrong shape for ``axis``,
            or an array is given without an axis (or vice versa).
    """
    if axis is None:
        if isinstance(scale, np.ndarray):
            raise ValueError("An array scale requires axis='row' or axis='col'")
        return csr_array((matrix.data / scale, matrix.indices, matrix.indptr), shape=matrix.shape)
    if not isinstance(scale, np.ndarray):
        raise TypeError(f"Axis-wise scaling requires an array scale, got {type(scale).__name__}")
    if axis == "row":
        if scale.shape != (matrix.shape[0],):
            raise ValueError(f"Row scale must have shape {(matrix.shape[0],)}, got {scale.shape}")
        row_of_value = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        return csr_array(
            (matrix.data / scale[row_of_value], matrix.indices, matrix.indptr),
            shape=matrix.shape,
        )
    if scale.shape != (matrix.shape[1],):
        raise ValueError(f"Column scale must have shape {(matrix.shape[1],)}, got {scale.shape}")
    return csr_array(
        (matrix.data / scale[matrix.indices], matrix.indices, matrix.indptr),
        shape=matrix.shape,
    )


def matrix_norm(matrix: SystemMatrix, kind: MatrixNormType) -> float:
    """Compute a matrix norm for either storage format.

    Dense matrices use ``np.linalg.norm`` (via ``calculate_matrix_norm``); CSR
    matrices use ``scipy.sparse.linalg.norm`` and are never densified.

    Raises:
        ValueError: If ``kind`` is NUCLEAR for a CSR matrix (not available in scipy).
    """
    if not isinstance(matrix, csr_array):
        return calculate_matrix_norm(matrix, kind)
    if kind is MatrixNormType.NUCLEAR:
        raise ValueError("Nuclear norm is not supported for sparse matrices")
    if kind is MatrixNormType.SPECTRAL:
        return _sparse_spectral_norm(matrix)
    return float(sparse_norm(matrix, ord=_SPARSE_NORM_ORDER[kind]))


def _sparse_spectral_norm(matrix: csr_array) -> float:
    """Largest singular value of a sparse matrix, deterministic across runs.

    ARPACK's default start vector is random, so ``scipy.sparse.linalg.norm``
    with ``ord=2`` is not reproducible. A fixed all-ones start vector gives the
    same result on every run; the dominant singular vector of a positive-
    definite stiffness matrix is not orthogonal to it.
    """
    start = np.ones(min(matrix.shape), dtype=np.float64)
    singular_values = svds(matrix, k=1, which="LM", return_singular_vectors=False, v0=start)
    return float(singular_values[0])


def _gershgorin_bound(matrix: SystemMatrix) -> float:
    """Gershgorin bound on the spectral radius: max(max row |.| sum, max col |.| sum).

    Equal to the larger of the 1-norm and inf-norm of the matrix, so it is
    computed through ``matrix_norm`` for both storage formats.
    """
    return max(matrix_norm(matrix, MatrixNormType.ONE), matrix_norm(matrix, MatrixNormType.INF))


def scale_factor(matrix: SystemMatrix, scaling: SystemScaling) -> float:
    """The scalar ``alpha`` a system with this matrix is divided by.

    ``NoScaling`` gives 1. ``GershgorinScaling`` gives the Gershgorin bound
    ``max(||A||_1, ||A||_inf)``, times ``sqrt(n)`` for ``DimensionFactor.SQRT_N``.
    """
    match scaling:
        case NoScaling():
            return 1.0
        case GershgorinScaling(dimension_factor=factor):
            return _gershgorin_bound(matrix) * _dimension_factor(factor, matrix.shape[0])


def _dimension_factor(factor: DimensionFactor, dimension: int) -> float:
    match factor:
        case DimensionFactor.ONE:
            return 1.0
        case DimensionFactor.SQRT_N:
            return compute_dim_scale(dimension)


def scale_matrix(matrix: SystemMatrix, alpha: float) -> SystemMatrix:
    """``matrix / alpha`` in the matrix's own format; a CSR matrix keeps its pattern."""
    if isinstance(matrix, csr_array):
        return scale_csr_values(matrix, alpha, None)
    return matrix / alpha


__all__ = [
    "ErrorTraceSamples",
    "ResidualTraceSamples",
    "matrix_norm",
    "scale_csr_values",
    "scale_factor",
    "scale_matrix",
]
