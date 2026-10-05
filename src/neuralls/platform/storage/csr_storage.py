"""Write-side storage for CSR system matrices (zarr group backend).

The on-disk schema lives in ``csr_layout``; this module only chooses the layout,
validates the write request, and writes the arrays into a zarr group.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import zarr
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array

from neuralls.platform.storage.csr_layout import (
    DATA_ARRAY,
    INDICES_ARRAY,
    INDPTR_ARRAY,
    SAMPLE_OFFSETS_ARRAY,
    SHAPE_ARRAY,
    PerSamplePack,
    SharedPatternPack,
    choose_layout,
    pack_per_sample,
    pack_shared_pattern,
)
from neuralls.platform.storage.manifest import DatasetArtifact
from neuralls.shared.types import DatasetFormat, LayoutType, MatrixFormat, SystemMatrix

_ZARR_FORMAT: DatasetFormat = "zarr"


class CsrAccumulator:
    """Collects COO or dense matrix samples as CSR, without densifying.

    COO components are converted with ``coo_array(...).tocsr()``, which sums
    duplicate ``(row, col)`` entries, so a repeated coordinate contributes the
    sum of its values. Dense input is converted with ``csr_array(matrix)``,
    which keeps only the nonzero entries.
    """

    def __init__(self) -> None:
        self._samples: list[csr_array] = []

    def append_sparse_components(
        self,
        *,
        indices: NDArray,
        values: NDArray,
        size: tuple[int, int],
        repeats: int,
    ) -> None:
        """Append one COO-described sample, repeated ``repeats`` times.

        Args:
            indices: ``(2, nnz)`` array of ``(row, col)`` coordinates.
            values: ``(nnz,)`` values, duplicates allowed (they are summed).
            size: ``(rows, cols)`` of the matrix.
            repeats: How many identical samples to append (>= 1).
        """
        matrix = coo_array((values, (indices[0], indices[1])), shape=size).tocsr()
        self._append(matrix, repeats)

    def append_dense_matrix(self, matrix: NDArray, repeats: int) -> None:
        """Append one dense-described sample, converted to CSR, ``repeats`` times."""
        self._append(csr_array(matrix), repeats)

    @property
    def samples(self) -> tuple[csr_array, ...]:
        """Every appended sample, in append order."""
        return tuple(self._samples)

    def _append(self, matrix: csr_array, repeats: int) -> None:
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        self._samples.extend([matrix] * repeats)


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


def _write_per_sample(group: zarr.Group, pack: PerSamplePack) -> None:
    group.create_array(INDPTR_ARRAY, data=pack.indptr)
    group.create_array(INDICES_ARRAY, data=pack.indices)
    group.create_array(DATA_ARRAY, data=pack.data)
    group.create_array(SAMPLE_OFFSETS_ARRAY, data=pack.sample_offsets)
    group.create_array(SHAPE_ARRAY, data=pack.shape)


def _write_shared_pattern(group: zarr.Group, pack: SharedPatternPack) -> None:
    group.create_array(INDPTR_ARRAY, data=pack.indptr)
    group.create_array(INDICES_ARRAY, data=pack.indices)
    group.create_array(DATA_ARRAY, data=pack.data)
    group.create_array(SHAPE_ARRAY, data=pack.shape)


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
    layout = choose_layout(samples)
    group = zarr.open_group(str(group_dir), mode="w")
    match layout:
        case LayoutType.SHARED_PATTERN:
            _write_shared_pattern(group, pack_shared_pattern(samples))
        case LayoutType.MANY_MATRICES:
            _write_per_sample(group, pack_per_sample(samples))
        case LayoutType.BROADCAST_SINGLE:
            raise ValueError("BROADCAST_SINGLE is not a CSR layout; CSR never produces it.")
    sample_count = len(samples)
    return DatasetArtifact(
        path=member_path,
        format=_ZARR_FORMAT,
        dtype="float64",
        shape=(sample_count,),
        n_matrix_samples=sample_count,
        layout=layout,
        matrix_format=MatrixFormat.CSR,
    )
