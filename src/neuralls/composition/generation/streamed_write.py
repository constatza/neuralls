"""Shared skeleton for the streamed dense and CSR write paths.

Both ``dense_streaming.py`` and ``csr_streaming.py`` plan a batch stream, fail fast if its
row count isn't exact, write into a store that must be released (not closed cleanly) on
failure, and record the same run-level manifest facts save for the layout. This module holds
exactly those pieces so neither streaming module has to repeat them.
"""

from __future__ import annotations

from neuralls.domain.generation.orchestration import BatchStream, open_batch_stream
from neuralls.domain.generation.ports import ArrayStore
from neuralls.domain.generation.scalar_aggregate import ScaleSummary
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.storage.dense_stream import StreamedManifestFacts
from neuralls.shared.types import LayoutType, MatrixFormat


def open_prepared_stream(
    source: SourceSpec,
    spec: DatasetSpec,
    *,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> BatchStream:
    """Plan the run and reject an inexact row count before any store is opened."""
    stream = open_batch_stream(
        source, spec, batch_size=spec.write_batch_size, matrix_format=matrix_format
    )
    stream.plan.require_exact()
    return stream


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
