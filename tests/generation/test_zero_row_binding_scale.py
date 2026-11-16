"""Scalar aggregation over bindings, with and without bindings that emit no rows.

The aggregator keeps the first observed binding's scalars and compares later ones with them.
Two cases follow from that contract:

- A binding the plan counts as contributing rows is observed exactly once, even if its
  strategy emits no batch for it.
- A binding whose counts are all zero is not planned at all, so it is never observed and
  leaves the dataset scale unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation import strategy_rows
from neuralls.domain.generation.interfaces import GeneratedSamples, SolvedRows
from neuralls.domain.generation.orchestration import BatchStream, open_batch_stream
from neuralls.domain.generation.scalar_aggregate import BindingScale, ScalarAggregator
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.shared.types import MatrixFormat
from tests import generation_support
from tests.fakes.archive_source import InMemoryArchiveSource

PER_BINDING_SAMPLES = 1
EMPTY_CALL_INDEX = 2
MATRIX_DIR = Path("archive/matrices")


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """The shared binding matrices ``A_0.txt``..``A_2.txt``."""
    source = InMemoryArchiveSource()
    for i, matrix in enumerate(generation_support.binding_matrices()):
        source.add(MATRIX_DIR / f"A_{i}.txt", matrix)
    return source


@pytest.fixture
def two_matrix_source() -> SourceSpec:
    """Source reading the first two matrices as two bindings."""
    return SourceSpec(
        matrix_path=str(MATRIX_DIR / "A_[01].txt"),
        sample_id_regex=generation_support.BINDING_SOURCE_REGEX,
    )


@pytest.fixture
def open_binding_stream(
    archive_source: InMemoryArchiveSource,
) -> Callable[[SourceSpec, dict[str, int]], BatchStream]:
    """Open a batch stream over the in-memory archive from a source and per-strategy counts."""

    def _open(source: SourceSpec, counts: dict[str, int]) -> BatchStream:
        spec = DatasetSpec(
            mixture=MixtureSpec(counts=counts, seed=generation_support.BINDING_SEED, shuffle=False)
        )
        return open_batch_stream(
            source,
            spec,
            batch_size=1 << 20,
            matrix_format=MatrixFormat.DENSE,
            readers=archive_source.readers(),
            archive_source=archive_source,
        )

    return _open


@pytest.fixture
def observed_scales(monkeypatch: pytest.MonkeyPatch) -> list[BindingScale]:
    """Record every BindingScale the aggregator observes, in call order."""
    observed: list[BindingScale] = []
    real_observe = ScalarAggregator.observe

    def _record(self: ScalarAggregator, binding: BindingScale) -> None:
        observed.append(binding)
        real_observe(self, binding)

    monkeypatch.setattr(ScalarAggregator, "observe", _record)
    return observed


@pytest.fixture
def second_run_emits_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the generated strategy return zero rows on its second call (binding 1)."""
    real_run_generation = strategy_rows.run_generation
    calls = [0]

    def _run(
        strategy_name: str, A: np.ndarray, *args: object, **kwargs: object
    ) -> GeneratedSamples:
        calls[0] += 1
        if calls[0] == EMPTY_CALL_INDEX:
            n = A.shape[0]
            return SolvedRows(
                matrix=A,
                rhs=np.empty((0, n), dtype=np.float64),
                solutions=np.empty((0, n), dtype=np.float64),
            )
        return real_run_generation(strategy_name, A, *args, **kwargs)

    monkeypatch.setattr(strategy_rows, "run_generation", _run)


def test_planned_binding_with_no_output_is_observed_once(
    two_matrix_source: SourceSpec,
    open_binding_stream: Callable[[SourceSpec, dict[str, int]], BatchStream],
    observed_scales: list[BindingScale],
    second_run_emits_no_rows: None,
) -> None:
    """Both bindings are planned with one sample each; binding 1 emits nothing but is observed."""
    stream = open_binding_stream(two_matrix_source, {"neutral_ones": 2 * PER_BINDING_SAMPLES})

    batches = list(stream.batches)

    assert [binding.binding_index for binding in stream.plan.bindings] == [0, 1]
    assert {batch.binding_index for batch in batches} == {0}
    assert len(observed_scales) == 2
    assert stream.scale.result().matrix_norm == observed_scales[0].matrix_norm_value


def test_zero_count_binding_is_not_planned_and_leaves_scale_unchanged(
    two_matrix_source: SourceSpec,
    open_binding_stream: Callable[[SourceSpec, dict[str, int]], BatchStream],
    observed_scales: list[BindingScale],
) -> None:
    """With one sample over two matrices, binding 0 gets zero: it is absent and not observed.

    The dataset scale then equals the scale of a run over the emitting matrix alone, recorded
    under that matrix's own sample id.
    """
    stream = open_binding_stream(two_matrix_source, {"neutral_ones": PER_BINDING_SAMPLES})
    batches = list(stream.batches)

    assert [binding.binding_index for binding in stream.plan.bindings] == [1]
    assert {batch.binding_index for batch in batches} == {1}
    assert len(observed_scales) == 1

    alone = open_binding_stream(
        SourceSpec(
            matrix_path=str(MATRIX_DIR / "A_1.txt"),
            sample_id_regex=two_matrix_source.sample_id_regex,
        ),
        {"neutral_ones": PER_BINDING_SAMPLES},
    )
    list(alone.batches)
    summary, alone_summary = stream.scale.result(), alone.scale.result()
    assert summary.matrix_scales.keys() == {1}
    assert (summary.matrix_norm, list(summary.matrix_scales.values())) == (
        alone_summary.matrix_norm,
        list(alone_summary.matrix_scales.values()),
    )
