"""Composition of a streamed CSR generation run: batches into a CSR stream and dense arrays into the store.

The dense arrays (rhs, solutions, row kind, matrix sample index, parameters) go through
SampleWriter into the same store the dense streamed path uses. The CSR matrix goes through
the CSR stream at the manifest's matrix address. Both are written into one staging directory
that is renamed to the dataset directory only after the manifest is written.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from scipy.sparse import csr_array

from neuralls.composition.generation.streamed_write import (
    manifest_facts,
    open_prepared_stream,
    release_after_failure,
)
from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.orchestration import BatchStream
from neuralls.domain.generation.sample_writer import SampleWriter
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.sparse_io.protocol import SparseStreamWriter
from neuralls.platform.storage.csr_storage import open_csr_matrix_stream
from neuralls.platform.storage.dense_stream import (
    csr_matrix_container,
    open_dense_array_store,
    save_csr_stream_manifest,
)
from neuralls.platform.storage.staged_commit import commit_staged_directory
from neuralls.shared.types import (
    DatasetFormat,
    LayoutType,
    MatrixFormat,
    SparsityPattern,
    SystemMatrix,
)

_ONE_SAMPLE: int = 1
"""Stored CSR samples for a single shared matrix: every row reuses the one stored matrix."""


def write_csr_streamed(
    source: SourceSpec,
    spec: DatasetSpec,
    dataset_dir: Path,
    dataset_format: DatasetFormat,
    *,
    sparsity_pattern: SparsityPattern,
) -> None:
    """Generate a CSR dataset batch by batch and commit it to ``dataset_dir``.

    Same staging and commit as ``write_dense_streamed``: ``dataset_dir`` is created only at
    commit, and a failed run leaves only the staging directory for the next run to replace.

    Args:
        source: Where the run reads its samples from.
        spec: Strategy budgets, RNG controls, normalization and batch size.
        dataset_dir: Final directory of the dataset; it is created only at commit.
        dataset_format: ``zarr`` or ``hdf5``.
        sparsity_pattern: ``[output].sparsity_pattern``. ``shared`` stores one pattern and
            raises if a matrix's pattern differs from the first; ``ragged`` stores each
            matrix's own pattern.

    Raises:
        ValueError: If the written rows differ from the plan.
        OSError: If the commit rename fails.
    """
    commit_staged_directory(
        dataset_dir,
        lambda staging: _write_into(source, spec, staging, dataset_format, sparsity_pattern),
    )


def _write_into(
    source: SourceSpec,
    spec: DatasetSpec,
    staging_dir: Path,
    dataset_format: DatasetFormat,
    sparsity_pattern: SparsityPattern,
) -> bool:
    stream = open_prepared_stream(source, spec, matrix_format=MatrixFormat.CSR)
    store = open_dense_array_store(staging_dir, dataset_format)
    try:
        csr_writer = open_csr_matrix_stream(
            dataset_format,
            csr_matrix_container(staging_dir, dataset_format),
            planned_samples=_planned_matrix_samples(stream),
            layout=_layout_for(sparsity_pattern),
        )
    except BaseException:
        release_after_failure(store)
        raise
    try:
        writer = SampleWriter(
            store,
            planned_rows=stream.plan.total_rows,
            matrix_for=None,
            single_matrix=stream.single_matrix,
        )
        _write_batches(stream, writer, csr_writer)
        artifacts = writer.finalize()
        summary = csr_writer.close()
    except BaseException:
        release_after_failure(store)
        _close_after_failure(csr_writer)
        raise
    save_csr_stream_manifest(
        staging_dir,
        dataset_format,
        artifacts,
        manifest_facts(spec, stream.scale.result(), summary.layout),
        summary,
    )
    return True


def _write_batches(
    stream: BatchStream,
    writer: SampleWriter,
    csr_writer: SparseStreamWriter,
) -> None:
    """Feed each batch to the dense writer and its matrix samples to the CSR stream.

    A single shared matrix is written once, at the first batch. Otherwise the stored
    samples are one per row, in generation order, which is what the buffered path stores.
    """
    matrix_written = False
    for batch in stream.batches:
        if not len(batch):
            continue
        writer.write_batch(batch)
        if not stream.single_matrix:
            csr_writer.write_batch(_csr_rows(stream, batch))
            continue
        if matrix_written:
            continue
        matrix_written = True
        csr_writer.write_batch([_csr_sample(stream, int(batch.matrix_sample_index[0]))])


def _csr_rows(stream: BatchStream, batch: SampleBatch) -> Sequence[csr_array]:
    return [_csr_sample(stream, int(sample_id)) for sample_id in batch.matrix_sample_index]


def _csr_sample(stream: BatchStream, sample_id: int) -> csr_array:
    matrix: SystemMatrix = stream.matrix_for(sample_id)
    if not isinstance(matrix, csr_array):
        raise TypeError("CSR batch streaming received a dense normalized matrix.")
    return matrix


def _planned_matrix_samples(stream: BatchStream) -> int:
    """Stored CSR samples: one for a shared matrix, one per row otherwise."""
    return _ONE_SAMPLE if stream.single_matrix else stream.plan.total_rows


def _layout_for(sparsity_pattern: SparsityPattern) -> LayoutType:
    """The stored layout is the configured sparsity pattern, never inferred from the data.

    ``shared`` stores one pattern with data of shape (N, nnz); a sample whose pattern differs
    is rejected by the stream when it is written. ``ragged`` stores every sample's own pattern.
    """
    match sparsity_pattern:
        case SparsityPattern.SHARED:
            return LayoutType.SHARED_PATTERN
        case SparsityPattern.RAGGED:
            return LayoutType.MANY_MATRICES


def _close_after_failure(csr_writer: SparseStreamWriter) -> None:
    """Close the CSR sink without masking the original error.

    A sink that was never fully written reports a sample-count mismatch on close; that is
    expected on this path, so the report is dropped in favour of the error being propagated.
    """
    try:
        csr_writer.close()
    except ValueError:
        return
