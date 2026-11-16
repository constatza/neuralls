"""Per-matrix-sample scaling cache: scale and measure a matrix once per sample id."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from neuralls.domain.normalization import matrix_norm as system_matrix_norm
from neuralls.domain.normalization import scale_factor, scale_matrix
from neuralls.shared.types import MatrixFormat, MatrixNormType, SystemMatrix

from .single_slot import SingleSlot
from .source_streams import MatrixSampleStream
from .specs import DatasetSpec


@dataclass(frozen=True, slots=True)
class _CachedMatrix:
    """One matrix sample after scaling, computed once per matrix sample id.

    Attributes:
        matrix_norm: Scaled system matrix ``A / alpha``, in the dataset's matrix format
            (dense ndarray or CSR; CSR is never densified).
        alpha: The scalar the raw matrix (and any archived RHS) is divided by.
        matrix_norm_value: Norm of the scaled matrix.
    """

    matrix_norm: SystemMatrix
    alpha: float
    matrix_norm_value: float


def _cached_matrix_loader(
    matrix_stream: MatrixSampleStream,
    spec: DatasetSpec,
    matrix_format: MatrixFormat,
) -> Callable[[int], _CachedMatrix]:
    """Return a loader that scales and measures one matrix sample per call.

    Only the most recent sample stays cached. Bindings are visited in order, so a matrix
    is needed only while its own binding runs, and this keeps one matrix in memory at a time.

    Args:
        matrix_stream: Source of the raw matrix samples.
        spec: Supplies the scaling and matrix norm settings.
        matrix_format: Storage format the raw matrices are loaded in.

    Returns:
        Callable mapping a matrix sample id to its cached scaled data.
    """
    slot = SingleSlot[int, _CachedMatrix]()

    def _load(sample_id: int) -> _CachedMatrix:
        raw_matrix = matrix_stream.load_sample(sample_id, matrix_format)
        if raw_matrix.shape[0] != raw_matrix.shape[1]:
            raise ValueError(f"Matrix sample {sample_id} must be square, got {raw_matrix.shape}")
        alpha = scale_factor(raw_matrix, spec.scaling)
        scaled = scale_matrix(raw_matrix, alpha)
        return _CachedMatrix(
            matrix_norm=scaled,
            alpha=alpha,
            matrix_norm_value=system_matrix_norm(scaled, MatrixNormType(spec.matrix_norm_type)),
        )

    def _get_matrix(sample_id: int) -> _CachedMatrix:
        return slot.get(sample_id, _load)

    return _get_matrix
