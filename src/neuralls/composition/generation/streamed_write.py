"""Shared skeleton for the streamed dense and CSR write paths.

Both ``dense_streaming.py`` and ``csr_streaming.py`` plan a batch stream, fail fast if its
row count isn't exact, write into a store that must be released (not closed cleanly) on
failure, and record the same run-level manifest facts save for the layout. This module holds
exactly those pieces so neither streaming module has to repeat them.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.orchestration import BatchStream, open_batch_stream
from neuralls.domain.generation.ports import ArrayStore
from neuralls.domain.generation.sample_writer import StrategyRowCount
from neuralls.domain.generation.scalar_aggregate import ScaleSummary
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.storage.dense_stream import StreamedManifestFacts
from neuralls.platform.storage.matrix_readers import read_dense_npy, read_matrix
from neuralls.platform.storage.staged_commit import commit_staged_directory
from neuralls.shared.types import LayoutType, MatrixFormat

_READERS = MatrixReaders(
    generic=partial(read_matrix, lazy=True),
    dense=partial(read_dense_npy, lazy=True),
)


def open_prepared_stream(
    source: SourceSpec,
    spec: DatasetSpec,
    *,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> BatchStream:
    """Plan the run and reject an inexact row count before any store is opened."""
    stream = open_batch_stream(
        source,
        spec,
        batch_size=spec.write_batch_size,
        matrix_format=matrix_format,
        readers=_READERS,
    )
    stream.plan.require_exact()
    return stream


def commit_with_rows(
    dataset_dir: Path, write: Callable[[Path], tuple[StrategyRowCount, ...]]
) -> tuple[StrategyRowCount, ...]:
    """Commit a staged dataset and return the rows per strategy its writer recorded.

    ``write`` always writes a dataset, so the commit is never declined; the rows it
    returns are those the committed arrays hold.
    """
    realized: tuple[StrategyRowCount, ...] = ()

    def staged(staging: Path) -> bool:
        nonlocal realized
        realized = write(staging)
        return True

    commit_staged_directory(dataset_dir, staged)
    return realized


def release_after_failure(store: ArrayStore) -> None:
    """Close the store without masking the original error.

    A store that was never fully written reports itself incomplete on close; that is
    expected here, so the report is dropped in favour of the error being propagated.
    """
    try:
        store.close()
    except ValueError:
        return


def manifest_facts(
    spec: DatasetSpec, scale_summary: ScaleSummary, layout: LayoutType
) -> StreamedManifestFacts:
    """Run-level manifest facts common to both streaming paths; only ``layout`` differs."""
    return StreamedManifestFacts(
        normalization_type=str(spec.normalize),
        matrix_norm=scale_summary.matrix_norm,
        matrix_norm_type=spec.matrix_norm_type,
        scale_metadata=scale_summary.scale_metadata,
        layout=layout,
    )
