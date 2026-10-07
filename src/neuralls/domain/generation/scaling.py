"""Matrix normalization and scale-metadata serialization for synthetic generation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.linalg import norm

from neuralls.domain.normalization import IScale
from neuralls.shared.types import ScaleMetadata, SystemMatrix


def _calculate_normalization_scale(
    A: np.ndarray,
    b: np.ndarray,
    normalize: str,
) -> float:
    """Calculate target RHS scale based on normalization strategy.

    Pure function.

    Args:
        A: System matrix
        b: Mother RHS vector
        normalize: Normalization strategy

    Returns:
        Target RHS scale for generation
    """
    if normalize == "rhs":
        n = A.shape[0]
        target_norm = float(norm(b))
        return target_norm / np.sqrt(n)
    return 1.0


def normalize_matrix_for_generation(
    matrix: SystemMatrix,
    normalize_type: Literal["none", "matrix", "rhs"],
    spectral_radius_bound: float | None,
) -> tuple[SystemMatrix, IScale | None, float]:
    """Normalize matrix for synthetic generation (pure function).

    CONTRACT:
        - Input: Raw matrix A
        - Output: Normalized matrix A_norm and optional scale metadata
        - Strategies receive normalized matrix and compute b_norm = A_norm @ x

    Args:
        matrix: Raw system matrix A, dense or CSR. CSR input stays CSR; it is
            never densified.
        normalize_type: Normalization strategy
            - "none": No normalization (identity)
            - "matrix": Scale by spectral_radius_bound * sqrt(d)
            - "rhs": Legacy, treated as "none" (RHS matching handled by caller)
        spectral_radius_bound: For matrix normalization (computed if None)

    Returns:
        Tuple of (normalized_matrix, scale_or_none, matrix_value_scale):
            - normalized_matrix: Matrix in normalized space
            - scale_or_none: MatrixScale object for "matrix", None for "none"/"rhs"
            - matrix_value_scale: Scalar that maps stored matrix values back to raw values
              (A_raw = A_stored * matrix_value_scale). 1.0 for "none"/"rhs".

    Examples:
        >>> A_norm, scale, value_scale = normalize_matrix_for_generation(A, "matrix", None)
        >>> isinstance(scale, MatrixScale)
        True
        >>> value_scale > 0
        True
    """
    from neuralls.domain.normalization import create_scale_from_config

    # No normalization: return defensive copy
    if normalize_type in ("none", "rhs"):
        return matrix.copy(), None, 1.0

    scale = create_scale_from_config(
        normalize_type=normalize_type,
        matrix=matrix,
        spectral_radius_bound=spectral_radius_bound,
    )
    assert scale is not None, f"Expected scale for {normalize_type}"
    matrix_norm = scale.scale_matrix(matrix)
    scale_params = serialize_scale_metadata(scale)
    if scale_params is None:
        raise TypeError(f"Expected scale metadata for {normalize_type}")
    spectral_radius = scale_params.get("spectral_radius_bound")
    dimension_scale = scale_params.get("dimension_scale")
    if not isinstance(spectral_radius, float) or not isinstance(dimension_scale, float):
        raise TypeError("Matrix normalization scale metadata must be scalar floats.")
    matrix_value_scale = spectral_radius * dimension_scale
    return matrix_norm, scale, matrix_value_scale


def serialize_scale_metadata(scale: IScale | None) -> ScaleMetadata | None:
    """Serialize supported scale objects into manifest metadata."""
    if scale is None:
        return None

    payload = scale.to_dict()
    metadata: ScaleMetadata = {}

    spectral_radius = payload.get("spectral_radius_bound")
    if isinstance(spectral_radius, float):
        metadata["spectral_radius_bound"] = spectral_radius

    dimension_scale = payload.get("dimension_scale")
    if isinstance(dimension_scale, float):
        metadata["dimension_scale"] = dimension_scale

    return metadata or None
