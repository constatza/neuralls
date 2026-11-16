"""Linear system scaling and validation for the comparison workflow."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_array

from neuralls.composition.comparison.models import (
    LinearSystem,
    ResolvedComparisonInput,
)
from neuralls.domain.normalization import scale_factor, scale_matrix
from neuralls.domain.solver.utils.validation import (
    validate_ax_equals_b,
    validate_matrix,
    validate_rhs_vector,
)
from neuralls.platform.config.models.comparison import RHS_SOURCE_MODEL_BY_KIND
from neuralls.platform.sparse_io.components import to_components
from neuralls.platform.sparse_io.torch_convert import to_torch_csr
from neuralls.platform.storage.manifest import PersistedMatrixScaling
from neuralls.shared.system_scaling import RhsScaling, SystemScaling, scaling_label
from neuralls.shared.types import ComparisonRhsSourceKind, SystemMatrix


class ScalingMismatchError(ValueError):
    """A stored matrix was scaled differently from what the case asks for.

    A stored matrix is never rescaled, so a case whose ``system_scaling`` differs from the
    dataset's would silently solve a differently scaled system than it declares.
    """

    def __init__(self, dataset_scaling: SystemScaling, case_scaling: SystemScaling) -> None:
        self.dataset_scaling = dataset_scaling
        self.case_scaling = case_scaling
        super().__init__(
            f"The dataset was generated with scaling {scaling_label(dataset_scaling)}, but the "
            f"case asks for system_scaling {scaling_label(case_scaling)}; regenerate the "
            "dataset or change the case."
        )


def _rhs_is_generated(rhs_source_kind: ComparisonRhsSourceKind | None) -> bool:
    """Whether the RHS is synthesized from its own parameters rather than derived from data.

    A generated RHS has a user-chosen magnitude, so a matrix-derived scale,
    fresh or persisted, must never touch it.
    """
    return rhs_source_kind is not None and RHS_SOURCE_MODEL_BY_KIND[rhs_source_kind].generated


def _scale_raw_system(
    matrix: SystemMatrix,
    rhs: np.ndarray,
    scaling: SystemScaling,
    *,
    rhs_source_kind: ComparisonRhsSourceKind | None,
) -> tuple[SystemMatrix, np.ndarray]:
    """Scale a raw matrix (no manifest) and its RHS by one freshly computed ``alpha``.

    The same ``alpha`` divides the matrix and (unless the RHS is generated) the RHS, never
    two independently derived values on the two sides of ``Ax = b``.
    """
    alpha = scale_factor(matrix, scaling)
    scaled = scale_matrix(matrix, alpha)
    if _rhs_is_generated(rhs_source_kind):
        return scaled, rhs
    return scaled, rhs / alpha


def _scale_stored_system(
    matrix: SystemMatrix,
    rhs: np.ndarray,
    persisted: PersistedMatrixScaling,
    *,
    rhs_source_kind: ComparisonRhsSourceKind | None,
) -> tuple[SystemMatrix, np.ndarray]:
    """Bring the RHS into the units of a matrix already scaled on disk.

    The matrix is never rescaled. The RHS is divided at most once, by this matrix's stored
    ``alpha`` (never a fresh recompute), and only when it is not already consistent with
    the matrix by construction:

    - DATASET-sourced RHS was scaled together with the matrix at generation — untouched.
    - RAW_LHS-sourced RHS was just derived as ``cfg.scale * matrix @ lhs`` from this exact
      (already scaled) matrix — untouched.
    - A generated RHS is independently configured — untouched.
    - RAW_RHS is a genuinely raw external vector with no associated x — divided by
      ``alpha`` to bring it into the matrix's units.
    """
    if rhs_source_kind is ComparisonRhsSourceKind.RAW_RHS:
        return matrix, rhs / persisted.alpha
    return matrix, rhs


def _scale_system(
    matrix: SystemMatrix,
    rhs: np.ndarray,
    system_scaling: SystemScaling,
    *,
    rhs_source_kind: ComparisonRhsSourceKind | None,
    matrix_scaling: PersistedMatrixScaling | None,
) -> tuple[SystemMatrix, np.ndarray]:
    """Apply the case's ``system_scaling`` to one matrix/RHS pair.

    Raises:
        ScalingMismatchError: If the matrix is stored with a different scaling.
    """
    if matrix_scaling is None:
        return _scale_raw_system(matrix, rhs, system_scaling, rhs_source_kind=rhs_source_kind)
    if matrix_scaling.scaling != system_scaling:
        raise ScalingMismatchError(matrix_scaling.scaling, system_scaling)
    return _scale_stored_system(matrix, rhs, matrix_scaling, rhs_source_kind=rhs_source_kind)


def _scale_rhs(rhs: np.ndarray, rhs_scaling: RhsScaling) -> np.ndarray:
    """Apply the case's per-system RHS scaling; a zero RHS is left unchanged."""
    match rhs_scaling:
        case RhsScaling.NONE:
            return rhs
        case RhsScaling.UNIT:
            rhs_norm = float(np.linalg.norm(rhs))
            return rhs if rhs_norm == 0.0 else rhs / rhs_norm


def prepare_resolved_linear_system(
    resolved_input: ResolvedComparisonInput,
    system_scaling: SystemScaling,
    rhs_scaling: RhsScaling,
) -> LinearSystem:
    """Scale and validate an already-resolved comparison input."""
    return _prepare_linear_system(
        resolved_input.matrix,
        resolved_input.rhs,
        system_scaling,
        rhs_scaling,
        rhs_source_kind=resolved_input.rhs_source_kind,
        matrix_scaling=resolved_input.matrix_scaling,
        lhs=resolved_input.lhs,
    )


def _prepare_linear_system(
    matrix: SystemMatrix,
    rhs: np.ndarray,
    system_scaling: SystemScaling,
    rhs_scaling: RhsScaling,
    *,
    rhs_source_kind: ComparisonRhsSourceKind | None,
    matrix_scaling: PersistedMatrixScaling | None,
    lhs: np.ndarray | None,
) -> LinearSystem:
    """Apply comparison scaling and validation to in-memory arrays.

    ``Ax = b`` is checked against ``lhs`` before the RHS scaling, which rescales the
    solution as well and would otherwise make the known ``lhs`` inconsistent.
    """
    matrix, rhs = _scale_system(
        matrix,
        rhs,
        system_scaling,
        rhs_source_kind=rhs_source_kind,
        matrix_scaling=matrix_scaling,
    )
    validate_matrix(matrix)
    if lhs is not None:
        validate_ax_equals_b(matrix, rhs, lhs)
    rhs = _scale_rhs(rhs, rhs_scaling)
    validate_rhs_vector(rhs, matrix)
    return LinearSystem(
        matrix=(
            to_torch_csr(to_components(matrix))
            if isinstance(matrix, csr_array)
            else torch.tensor(matrix, dtype=torch.float64)
        ),
        rhs=torch.tensor(rhs, dtype=torch.float64),
    )
