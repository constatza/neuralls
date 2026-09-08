"""Linear system loading, normalization, and diagnostics for the comparison workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from neuralls.composition.comparison.models import (
    ComparisonPaths,
    LinearSystem,
    ResolvedComparisonInput,
)
from neuralls.domain.linalg import compute_condition_number
from neuralls.domain.normalization import IScale, create_scale_from_config, load_scale_from_metadata
from neuralls.domain.solver.utils.validation import (
    validate_ax_equals_b,
    validate_matrix,
    validate_rhs_vector,
)
from neuralls.platform.storage.comparison import load_system_arrays
from neuralls.platform.storage.manifest import DatasetNormalization
from neuralls.shared.types import ComparisonRhsSourceKind

# RHS sources that are independently configured (a user-chosen magnitude, or an
# explicit sparse pattern) rather than derived from any raw/normalized matrix.
# These must never be touched by a matrix-derived scale, fresh or persisted.
_SYNTHETIC_RHS_KINDS = frozenset({ComparisonRhsSourceKind.GAUSSIAN, ComparisonRhsSourceKind.SPARSE})


def _self_normalize_rhs(rhs: np.ndarray) -> np.ndarray:
    """Scale rhs by its own norm, independent of any matrix-derived scale."""
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return rhs
    return rhs / rhs_norm


def _apply_fresh_matrix_scale(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    rhs_source_kind: ComparisonRhsSourceKind | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a genuinely raw matrix, computing one shared scale for both sides.

    The same freshly-computed scale is applied to the matrix and (unless the
    RHS is independently configured) to the RHS — never two independently
    derived scale values combined on the two sides of ``Ax = b``.
    """
    scale = create_scale_from_config("matrix", matrix)
    if scale is None:
        return matrix, rhs
    if not isinstance(scale, IScale):
        raise TypeError(f"Expected IScale, got {type(scale).__name__}")
    scaled_matrix = scale.scale_matrix(matrix)
    if rhs_source_kind in _SYNTHETIC_RHS_KINDS:
        return scaled_matrix, rhs
    return scaled_matrix, scale.scale_rhs(rhs)


def _apply_persisted_matrix_scale(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    rhs_source_kind: ComparisonRhsSourceKind | None,
    matrix_normalization: DatasetNormalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize against a matrix already known to be normalized on disk.

    The matrix is never rescaled. The RHS is scaled at most once, using the
    dataset's *persisted* scale (never a fresh recompute), and only when it
    hasn't already been made consistent with the matrix by construction:

    - DATASET-sourced RHS was normalized together with the matrix at
      generation time — untouched.
    - RAW_LHS-sourced RHS was just derived as ``cfg.scale * matrix @ lhs``
      from this exact (already-normalized) matrix — untouched.
    - GAUSSIAN/SPARSE RHS is independently configured — untouched.
    - RAW_RHS is a genuinely raw external vector with no associated x — scaled
      exactly once by the persisted scale to bring it into the matrix's units.
    """
    if rhs_source_kind is ComparisonRhsSourceKind.RAW_RHS:
        persisted_scale = load_scale_from_metadata(
            matrix_normalization.type, matrix_normalization.scale
        )
        if persisted_scale is None:
            return matrix, rhs
        return matrix, persisted_scale.scale_rhs(rhs)
    return matrix, rhs


@dataclass(frozen=True)
class _NormalizationRequest:
    """One matrix/RHS pair plus everything a normalization mode may consult."""

    matrix: np.ndarray
    rhs: np.ndarray
    rhs_source_kind: ComparisonRhsSourceKind | None
    matrix_normalization: DatasetNormalization | None


type NormalizationMode = Callable[[_NormalizationRequest], tuple[np.ndarray, np.ndarray]]


def _normalize_none(request: _NormalizationRequest) -> tuple[np.ndarray, np.ndarray]:
    """Leave both sides exactly as loaded."""
    return request.matrix, request.rhs


def _normalize_matrix(request: _NormalizationRequest) -> tuple[np.ndarray, np.ndarray]:
    """Bring the system into the matrix's units with one matrix-derived scale."""
    persisted = request.matrix_normalization
    if persisted is not None and persisted.type == "matrix":
        return _apply_persisted_matrix_scale(
            request.matrix,
            request.rhs,
            rhs_source_kind=request.rhs_source_kind,
            matrix_normalization=persisted,
        )
    return _apply_fresh_matrix_scale(
        request.matrix, request.rhs, rhs_source_kind=request.rhs_source_kind
    )


def _normalize_rhs(request: _NormalizationRequest) -> tuple[np.ndarray, np.ndarray]:
    """Scale the RHS by its own norm, leaving the matrix untouched."""
    return request.matrix, _self_normalize_rhs(request.rhs)


def _normalize_both(request: _NormalizationRequest) -> tuple[np.ndarray, np.ndarray]:
    """Apply the matrix mode, then self-normalize the RHS it produced."""
    matrix, rhs = _normalize_matrix(request)
    return matrix, _self_normalize_rhs(rhs)


# Registry mapping config-level normalize_system names to their mode functions.
_NORMALIZATION_REGISTRY: dict[str, NormalizationMode] = {
    "none": _normalize_none,
    "matrix": _normalize_matrix,
    "rhs": _normalize_rhs,
    "both": _normalize_both,
}


def register_normalization_mode(name: str, mode: NormalizationMode) -> None:
    """Register a comparison-time normalization mode.

    Args:
        name: Config-level ``normalize_system`` value.
        mode: Callable turning a request into the scaled ``(matrix, rhs)`` pair.
    """
    _NORMALIZATION_REGISTRY[name] = mode


def _require_supported_persisted_normalization(
    matrix_normalization: DatasetNormalization | None,
) -> None:
    """Reject a matrix whose on-disk normalization scheme is no longer supported.

    Raises:
        ValueError: If the persisted normalization type is neither
            ``"none"`` nor ``"matrix"``.
    """
    if matrix_normalization is None or matrix_normalization.type in ("none", "matrix"):
        return
    raise ValueError(
        f"Dataset's persisted normalization type {matrix_normalization.type!r} is no "
        "longer supported (only 'none'/'matrix' are) — regenerate the dataset with "
        "normalize_type='matrix'."
    )


def _normalize_linear_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    normalize_system: str,
    *,
    rhs_source_kind: ComparisonRhsSourceKind | None = None,
    matrix_normalization: DatasetNormalization | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply comparison-time normalization to matrix and RHS.

    Never independently recomputes a scale for data that is already known to
    be normalized — see ``_apply_persisted_matrix_scale``. A single scale is
    used for both sides of ``Ax = b`` in every mode; RHS is scaled at most
    once, never zero-or-two times ambiguously.

    Args:
        matrix: System matrix A.
        rhs: Right-hand side vector b.
        normalize_system: Scaling mode — one of ``"none"``, ``"matrix"``,
            ``"rhs"``, ``"both"``.
        rhs_source_kind: Provenance of the RHS, used to decide whether it may
            be touched by a matrix-derived scale at all.
        matrix_normalization: Persisted normalization metadata for the
            dataset the matrix was loaded from, or None for a raw/external
            matrix with no manifest.

    Returns:
        Scaled (matrix, rhs) pair.

    Raises:
        ValueError: If normalize_system is not a registered mode, or if the
            matrix's persisted normalization type is no longer supported.
        TypeError: If create_scale_from_config returns an unexpected type.
    """
    _require_supported_persisted_normalization(matrix_normalization)
    mode = _NORMALIZATION_REGISTRY.get(normalize_system)
    if mode is None:
        raise ValueError(
            f"Unsupported normalize_system value: {normalize_system!r}. "
            f"Expected one of {', '.join(_NORMALIZATION_REGISTRY)}."
        )
    return mode(
        _NormalizationRequest(
            matrix=matrix,
            rhs=rhs,
            rhs_source_kind=rhs_source_kind,
            matrix_normalization=matrix_normalization,
        )
    )


def _load_linear_system(
    paths: ComparisonPaths,
    *,
    rhs_sample_index: int,
    matrix_index: int,
    normalize_system: str,
    resolved_input: ResolvedComparisonInput | None = None,
) -> LinearSystem:
    """Load and validate a linear system from disk.

    Args:
        paths: Resolved comparison paths with matrix and rhs file locations.
        rhs_sample_index: Row index to select from a multi-RHS file; -1 selects row 0.
        matrix_index: Sample index to select from a multi-matrix dataset directory.
        normalize_system: Scaling mode — one of ``"none"``, ``"matrix"``,
            ``"rhs"``, ``"both"``.

    Returns:
        Validated and scaled LinearSystem.

    Raises:
        ValueError: If validation fails (wrong shape, NaN values, incompatible
            dimensions) or the Ax=b invariant is violated when a known true
            solution is available.
        FileNotFoundError: If matrix or rhs files don't exist.
    """
    if resolved_input is None:
        A, b = load_system_arrays(
            paths.matrix, paths.rhs, rhs_sample_index=rhs_sample_index, matrix_index=matrix_index
        )
    else:
        A, b = resolved_input.matrix, resolved_input.rhs
    A, b = _normalize_linear_system(
        A,
        b,
        normalize_system,
        rhs_source_kind=resolved_input.rhs_source_kind if resolved_input else None,
        matrix_normalization=resolved_input.matrix_normalization if resolved_input else None,
    )
    validate_matrix(A)
    validate_rhs_vector(b, A)
    if resolved_input is not None and resolved_input.lhs is not None:
        validate_ax_equals_b(A, b, resolved_input.lhs)
    return LinearSystem(
        matrix=torch.tensor(A, dtype=torch.float64),
        rhs=torch.tensor(b, dtype=torch.float64),
    )


def _log_matrix_condition_number(
    matrix: np.ndarray,
    *,
    matrix_path: Path,
    display_name: str | None,
) -> float:
    """Log the matrix condition number once per comparison.

    Args:
        matrix: System matrix to evaluate.
        matrix_path: Source path (used for log labels).
        display_name: Optional human-readable comparison label.

    Returns:
        The computed condition number, or NaN if it could not be computed.
    """
    label = display_name or matrix_path.stem or matrix_path.name
    try:
        value = float(compute_condition_number(matrix))
    except np.linalg.LinAlgError as exc:
        logger.warning(
            f"Matrix condition number unavailable: comparison={label} "
            f"matrix={matrix_path.name} error={exc}"
        )
        return float("nan")
    logger.info(
        f"Matrix condition number: comparison={label} matrix={matrix_path.name} value={value:.4e}"
    )
    return value
