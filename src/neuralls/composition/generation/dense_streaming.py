"""Composition of a streamed dense generation run: batches into SampleWriter into the store."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from numpy.typing import NDArray
from scipy.sparse import csr_array

from neuralls.composition.generation.finalize import commit_staged_directory
from neuralls.domain.generation.orchestration import open_batch_stream
from neuralls.domain.generation.ports import ArrayStore
from neuralls.domain.generation.sample_writer import MatrixLookup, SampleWriter
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.storage.dense_stream import (
    StreamedManifestFacts,
    open_dense_array_store,
    save_dense_stream_manifest,
)
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
    stream = open_batch_stream(source, spec, batch_size=spec.write_batch_size)
    stream.plan.require_exact()
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
    summary = stream.scale.result()
    save_dense_stream_manifest(
        staging_dir,
        dataset_format,
        artifacts,
        StreamedManifestFacts(
            normalization_type=str(spec.normalize),
            matrix_norm=summary.matrix_norm,
            matrix_norm_type=spec.matrix_norm_type,
            scale_metadata=summary.scale_metadata,
            layout=LayoutType.BROADCAST_SINGLE
            if stream.single_matrix
            else LayoutType.MANY_MATRICES,
        ),
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


def release_after_failure(store: ArrayStore) -> None:
    """Close the store without masking the original error.

    A store that was never fully written reports itself incomplete on close; that is
    expected here, so the report is dropped in favour of the error being propagated.
    """
    try:
        store.close()
    except ValueError:
        return
