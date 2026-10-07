"""Per-matrix-sample normalization cache: normalize and measure a matrix once per sample id."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from neuralls.domain.normalization import IScale
from neuralls.domain.normalization import matrix_norm as system_matrix_norm
from neuralls.shared.types import MatrixFormat, MatrixNormType, ScaleMetadata, SystemMatrix

from .helpers import normalize_matrix_for_generation, serialize_scale_metadata
from .source_streams import MatrixSampleStream
from .specs import DatasetSpec


@dataclass(frozen=True)
class _CachedMatrix:
    """Immutable cached matrix data for generation.

    Stores all derived matrices and scaling information computed once
    per unique matrix_sample_id, avoiding redundant computation.

    Attributes:
        matrix_norm: Normalized system matrix, in the dataset's matrix format
            (dense ndarray or CSR; CSR is never densified)
        scale: IScale object or None (scaling strategy applied)
        matrix_norm_value: Computed matrix norm value
        matrix_value_scale: Scaling factor applied
        scale_params: Dictionary of scale parameters or None
    """

    matrix_norm: SystemMatrix
    scale: IScale | None
    matrix_norm_value: float
    matrix_value_scale: float
    scale_params: ScaleMetadata | None


def _cached_matrix_loader(
    matrix_stream: MatrixSampleStream,
    spec: DatasetSpec,
    matrix_format: MatrixFormat,
) -> Callable[[int], _CachedMatrix]:
    """Return a loader that normalizes and measures one matrix sample per call.

    Only the most recent sample stays cached. Bindings are visited in order, so a matrix
    is needed only while its own binding runs, and this keeps one matrix in memory at a time.

    Args:
        matrix_stream: Source of the raw matrix samples.
        spec: Supplies the normalization and matrix norm settings.
        matrix_format: Storage format the raw matrices are loaded in.

    Returns:
        Callable mapping a matrix sample id to its cached normalized data.
    """

    @lru_cache(maxsize=1)
    def _get_matrix(sample_id: int) -> _CachedMatrix:
        raw_matrix = matrix_stream.load_sample(sample_id, matrix_format)
        if raw_matrix.shape[0] != raw_matrix.shape[1]:
            raise ValueError(f"Matrix sample {sample_id} must be square, got {raw_matrix.shape}")
        matrix_norm, scale, matrix_value_scale = normalize_matrix_for_generation(
            raw_matrix,
            spec.normalize,
            spectral_radius_bound=None,
        )
        matrix_norm_value = system_matrix_norm(matrix_norm, MatrixNormType(spec.matrix_norm_type))
        scale_params = serialize_scale_metadata(scale)
        return _CachedMatrix(
            matrix_norm=matrix_norm,
            scale=scale,
            matrix_norm_value=matrix_norm_value,
            matrix_value_scale=matrix_value_scale,
            scale_params=scale_params,
        )

    return _get_matrix
