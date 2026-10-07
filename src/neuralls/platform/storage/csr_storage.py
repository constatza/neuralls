"""Write-side storage for CSR system matrices.

The on-disk schema lives in ``csr_layout``. Writing goes through the sparse
backend registry, so the accumulator can stage either a zarr group or an hdf5
group; this module validates the request and converts samples to components.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import to_components
from neuralls.platform.sparse_io.protocol import (
    SparseLocation,
    SparseReader,
    SparseStreamWriter,
    SparseWriter,
)
from neuralls.platform.sparse_io.registry import backend_for, open_stream_writer
from neuralls.platform.storage.manifest import DatasetArtifact
from neuralls.shared.types import DatasetFormat, LayoutType, MatrixFormat, SystemMatrix

_ZARR_FORMAT: DatasetFormat = "zarr"
_CSR_STREAM_FORMATS: frozenset[DatasetFormat] = frozenset({"zarr", "hdf5"})


def _require_csr_samples(matrices: Sequence[SystemMatrix]) -> tuple[csr_array, ...]:
    """Return the samples as CSR, rejecting empty input and any dense or mixed sample."""
    if not matrices:
        raise ValueError("A CSR dataset requires at least one matrix sample")
    dense_count = sum(1 for matrix in matrices if not isinstance(matrix, csr_array))
    if dense_count:
        raise ValueError(
            f"A CSR dataset requires every sample to be csr_array; got {dense_count} "
            f"non-CSR sample(s) out of {len(matrices)}. Mixed or dense formats are not stored "
            "in one CSR dataset."
        )
    return tuple(matrix for matrix in matrices if isinstance(matrix, csr_array))


def _csr_artifact(layout: LayoutType, sample_count: int, member_path: str) -> DatasetArtifact:
    return DatasetArtifact(
        path=member_path,
        format=_ZARR_FORMAT,
        dtype="float64",
        shape=(sample_count,),
        n_matrix_samples=sample_count,
        layout=layout,
        matrix_format=MatrixFormat.CSR,
    )


def _zarr_writer() -> SparseWriter:
    return backend_for(_ZARR_FORMAT)[1]


def _zarr_reader() -> SparseReader:
    return backend_for(_ZARR_FORMAT)[0]


def write_csr_matrix_group(
    dataset_format: DatasetFormat,
    group_dir: Path,
    matrices: Sequence[SystemMatrix],
    *,
    member_path: str,
) -> DatasetArtifact:
    """Write CSR matrix samples as a zarr group and describe it for the manifest.

    This is the single write-side entry point for CSR storage. It rejects every
    request that the layout cannot represent before touching the disk.

    Args:
        dataset_format: Dataset storage family from ``[output].dataset_format``.
            Only ``"zarr"`` is supported for CSR.
        group_dir: Directory of the zarr group to create (overwritten if present).
        matrices: One CSR sample per logical matrix. Every entry must be a
            ``csr_array``.
        member_path: Manifest path of the group, relative to the dataset root.

    Returns:
        The manifest descriptor with ``matrix_format=CSR`` and the chosen layout.
        Its ``shape`` is ``(N,)``, the number of matrix samples.

    Raises:
        ValueError: If the format is not zarr, the samples are empty, or any
            sample is dense (mixed formats).
    """
    if dataset_format != _ZARR_FORMAT:
        raise ValueError(
            f"CSR storage is zarr-only for now; dataset_format={dataset_format!r} is not "
            "supported with matrix_format='csr'. Use dataset_format='zarr'."
        )
    samples = _require_csr_samples(matrices)
    summary = _zarr_writer().write(
        SparseLocation(path=group_dir), [to_components(sample) for sample in samples]
    )
    return _csr_artifact(summary.layout, summary.sample_count, member_path)


def describe_csr_matrix_group(group_dir: Path, *, member_path: str) -> DatasetArtifact:
    """Describe an already-written CSR zarr group for the manifest.

    The layout is read back from the group's members: only the MANY_MATRICES
    layout stores per-sample offsets, so their presence identifies it. This lets
    the dataset writer record the layout that ``write_csr_matrix_group`` chose
    without needing the samples again.

    Args:
        group_dir: Directory of a CSR zarr group written by ``write_csr_matrix_group``.
        member_path: Manifest path of the group, relative to the dataset root.

    Returns:
        The manifest descriptor with ``matrix_format=CSR``, the layout and sample count.
    """
    summary = _zarr_reader().summary(SparseLocation(path=group_dir))
    return _csr_artifact(summary.layout, summary.sample_count, member_path)


def open_csr_matrix_stream(
    dataset_format: DatasetFormat,
    container: Path,
    *,
    planned_samples: int,
    layout: LayoutType,
) -> SparseStreamWriter:
    """Open an incremental CSR matrix container that receives samples batch by batch.

    This is the streamed counterpart of ``write_csr_matrix_group`` and stores the same
    members. The caller chooses ``layout`` (the buffered path picks it with
    ``choose_layout`` over every sample, which a stream cannot see in advance) and passes
    the exact sample count. After ``close`` returns, the caller records the matrix
    descriptor from the returned summary, or from ``describe_csr_matrix_group`` for zarr.

    Args:
        dataset_format: ``"zarr"`` (``container`` is the group directory) or ``"hdf5"``
            (``container`` is the file; the CSR members go in its ``matrix`` group, which is
            where the manifest addresses them).
        container: Group directory for zarr, or HDF5 file for hdf5. Replaced if present.
        planned_samples: Exact number of samples that will be written. Must be at least 1.
        layout: ``MANY_MATRICES`` for per-sample patterns, or ``SHARED_PATTERN`` when every
            sample has the first sample's pattern. A mismatching sample raises on write.

    Returns:
        A writer with ``write_batch(samples)`` and ``close() -> SparseSummary``.

    Raises:
        ValueError: If the format is not zarr or hdf5, or the plan is invalid.
    """
    if dataset_format not in _CSR_STREAM_FORMATS:
        raise ValueError(
            f"CSR storage supports zarr and hdf5; dataset_format={dataset_format!r} is not "
            "supported with matrix_format='csr'."
        )
    return open_stream_writer(
        dataset_format,
        SparseLocation(path=container),
        planned_samples=planned_samples,
        layout=layout,
    )
