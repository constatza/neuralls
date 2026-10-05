"""Normalization utilities for the domain layer.

Canonicalized on matrix-based (Gershgorin-bound) normalization only. Provides
a small, composable API: an ABC interface for the scale strategy, a frozen
dataclass implementation, and pure factory functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse.linalg import svds

from neuralls.domain.linalg import calculate_matrix_norm, compute_dim_scale
from neuralls.shared.types import MatrixNormType, SystemMatrix

# Sparse norm orders passed to scipy.sparse.linalg.norm. Nuclear is absent
# because scipy has no sparse implementation of it.
_SPARSE_NORM_ORDER: Final[dict[MatrixNormType, Literal["fro"] | float]] = {
    MatrixNormType.FROBENIUS: "fro",
    MatrixNormType.ONE: 1,
    MatrixNormType.INF: np.inf,
}

# =============================================================================
# ABC Interfaces
# =============================================================================


class IScale(ABC):
    """Interface for scaling strategies.

    Encapsulates all scaling logic for a normalization strategy.
    """

    @abstractmethod
    def scale_matrix(self, matrix: SystemMatrix) -> SystemMatrix:
        """Scale the system matrix A, preserving its storage format."""
        ...

    @abstractmethod
    def scale_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Scale a right-hand side vector b."""
        ...

    @abstractmethod
    def to_dict(self) -> Mapping[str, float | list[float]]:
        """Serialize scale parameters to dictionary.

        Returns:
            Mapping of scale parameters suitable for saving to dataset manifests.
        """
        ...


# =============================================================================
# Frozen Dataclasses: Scaling Strategy
# =============================================================================


@dataclass(frozen=True)
class MatrixScale(IScale):
    """Matrix normalization: scale by spectral radius bound * sqrt(d).

    Transformation summary:
        A' = A / (spectral_radius_bound * sqrt(d))
        b' = b / (spectral_radius_bound * sqrt(d))

    The normalized system A' @ x = b' has the same solution as A @ x = b.
    """

    spectral_radius_bound: float
    dimension_scale: float

    @property
    def composite_scale(self) -> float:
        """Combined scale factor."""
        return self.spectral_radius_bound * self.dimension_scale

    def scale_matrix(self, matrix: SystemMatrix) -> SystemMatrix:
        if isinstance(matrix, csr_array):
            return scale_csr_values(matrix, self.composite_scale, None)
        return matrix / self.composite_scale

    def scale_rhs(self, rhs: np.ndarray) -> np.ndarray:
        return rhs / self.composite_scale

    def to_dict(self) -> dict[str, float]:
        """Serialize scale parameters to dictionary."""
        return {
            "spectral_radius_bound": self.spectral_radius_bound,
            "dimension_scale": self.dimension_scale,
        }


# =============================================================================
# Frozen Dataclasses: Trace Samples
# =============================================================================


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    """Gershgorin spectral radius bound: max(max row |.| sum, max col |.| sum).

    Equal to the larger of the 1-norm and inf-norm of the matrix, so it is
    computed through ``matrix_norm`` for both storage formats.
    """
    return max(matrix_norm(matrix, MatrixNormType.ONE), matrix_norm(matrix, MatrixNormType.INF))


# =============================================================================
# Pure Functions: Scale Factories
# =============================================================================


def _create_matrix_scale(
    matrix: SystemMatrix,
    spectral_radius_bound: float | None = None,
    **_kwargs: Any,
) -> MatrixScale:
    """Create matrix normalization scale.

    Args:
        matrix: System matrix A
        spectral_radius_bound: Optional spectral radius bound (computed if None)

    Returns:
        MatrixScale object
    """
    dimension = matrix.shape[0]
    if spectral_radius_bound is None:
        spectral_radius_bound = _gershgorin_bound(matrix)
    dimension_scale = compute_dim_scale(dimension)
    return MatrixScale(
        spectral_radius_bound=spectral_radius_bound,
        dimension_scale=dimension_scale,
    )


def load_scale_from_metadata(
    normalize_type: str,
    metadata: dict,
) -> IScale | None:
    """Reconstruct scale object from saved metadata.

    Args:
        normalize_type: Type of normalization ("none" or "matrix")
        metadata: Dictionary containing scale parameters

    Returns:
        Reconstructed scale object, or None if normalize_type is "none"

    Raises:
        ValueError: If normalize_type is invalid or required parameters are missing
    """
    if normalize_type == "none":
        return None

    if normalize_type == "matrix":
        if "spectral_radius_bound" not in metadata or "dimension_scale" not in metadata:
            raise ValueError(
                "Matrix normalization requires 'spectral_radius_bound' and 'dimension_scale' in metadata"
            )
        return MatrixScale(
            spectral_radius_bound=metadata["spectral_radius_bound"],
            dimension_scale=metadata["dimension_scale"],
        )

    raise ValueError(f"Invalid normalize_type: {normalize_type}. Must be one of: none, matrix")


# =============================================================================
# Pure Functions: Config-based Scale Creation
# =============================================================================


_SCALE_CREATORS = {
    "matrix": _create_matrix_scale,
}


def create_scale_from_config(
    normalize_type: Literal["none", "matrix"],
    matrix: SystemMatrix,
    *,
    spectral_radius_bound: float | None = None,
) -> IScale | None:
    """Create normalization scale object from configuration.

    Args:
        normalize_type: Type of normalization ("none" or "matrix")
        matrix: System matrix A
        spectral_radius_bound: For matrix normalization. If None, computed from matrix.

    Returns:
        Scale object, or None (for "none" type).

    Raises:
        ValueError: If normalize_type is invalid.

    Examples:
        >>> scale = create_scale_from_config("matrix", A)
        >>> isinstance(scale, MatrixScale)
        True
    """
    if normalize_type == "none":
        return None

    creator = _SCALE_CREATORS.get(normalize_type)
    if creator is None:
        raise ValueError(f"Invalid normalize_type: {normalize_type}. Must be one of: none, matrix")

    return creator(matrix=matrix, spectral_radius_bound=spectral_radius_bound)


__all__ = [
    "ErrorTraceSamples",
    # Interfaces
    "IScale",
    # Scale
    "MatrixScale",
    # Traces
    "ResidualTraceSamples",
    # Public API
    "create_scale_from_config",
    "load_scale_from_metadata",
    "matrix_norm",
    "scale_csr_values",
]
