"""Composition of a streamed dense generation run: batches into SampleWriter into the store."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from numpy.typing import NDArray
from scipy.sparse import csr_array

from neuralls.composition.generation.streamed_write import (
    manifest_facts,
    open_prepared_stream,
    release_after_failure,
)
from neuralls.domain.generation.sample_writer import MatrixLookup, SampleWriter
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.storage.dense_stream import (
    open_dense_array_store,
    save_dense_stream_manifest,
)
from neuralls.platform.storage.staged_commit import commit_staged_directory
from neuralls.shared.types import DatasetFormat, LayoutType, SystemMatrix


def write_dense_streamed(
    source: SourceSpec,
    spec: DatasetSpec,
    dataset_dir: Path,
    dataset_format: DatasetFormat,
) -> None:
    """Generate a dense dataset batch by batch and commit it to ``dataset_dir``.

    Memory is bounded by ``spec.write_batch_size`` rows plus one matrix, not by the dataset.
    Everything is written into a staging directory that is renamed to ``dataset_dir`` only
    after the manifest is written, so ``dataset_dir`` never holds a partial dataset. If
    generation fails part-way, the error propagates and the staging directory is left for
    the next run to replace.

    Args:
        source: Where the run reads its samples from.
        spec: Strategy budgets, RNG controls, normalization and batch size.
        dataset_dir: Final directory of the dataset; it is created only at commit.
        dataset_format: ``zarr`` or ``hdf5``.


    Raises:
        ValueError: If the written rows differ from the plan.
        OSError: If the commit rename fails.
    """
    commit_staged_directory(
        dataset_dir,
        lambda staging: _write_into(source, spec, staging, dataset_format),
    )


def _write_into(
    source: SourceSpec,
    spec: DatasetSpec,
    staging_dir: Path,
    dataset_format: DatasetFormat,
) -> bool:
    stream = open_prepared_stream(source, spec)
    store = open_dense_array_store(staging_dir, dataset_format)
    try:
        writer = SampleWriter(
            store,
            planned_rows=stream.plan.total_rows,
            matrix_for=_dense_lookup(stream.matrix_for),
            single_matrix=stream.single_matrix,
        )
        for batch in stream.batches:
            writer.write_batch(batch)
        artifacts = writer.finalize()
    except BaseException:
        release_after_failure(store)
        raise
    layout = LayoutType.BROADCAST_SINGLE if stream.single_matrix else LayoutType.MANY_MATRICES
    save_dense_stream_manifest(
        staging_dir,
        dataset_format,
        artifacts,
        manifest_facts(spec, stream.scale.result(), layout),
    )
    return True


def _dense_lookup(matrix_for: Callable[[int], SystemMatrix]) -> MatrixLookup:
    """Narrow the stream's matrix lookup to dense matrices, which the dense stores expect."""

    def dense_matrix_for(sample_id: int) -> NDArray:
        matrix = matrix_for(sample_id)
        if isinstance(matrix, csr_array):
            raise TypeError("Dense batch streaming received a sparse normalized matrix.")
        return matrix

    return dense_matrix_for
