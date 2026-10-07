"""Explicit solution rows fed to a solution-archive strategy must come from a non-empty file."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest

from neuralls.domain.generation.batch_plan import BindingAllocation
from neuralls.domain.generation.orchestration import (
    OpenedStreams,
    _GenerationRunContext,
    _solution_archive_rows,
)
from neuralls.domain.generation.source_streams import MatrixSampleStream, VectorSampleStream
from neuralls.shared.types import GenerationStrategyKind

_STRATEGY = GenerationStrategyKind.SOLUTION_ARCHIVE.value


@dataclass(frozen=True)
class _EmptySolutionStream:
    """Solution stream over a file that holds no samples."""

    sample_ids: tuple[int, ...] = ()

    def load_sample(self, sample_id: int) -> object:
        raise AssertionError("an empty file has no sample to load")


@pytest.fixture
def empty_solution_context() -> _GenerationRunContext:
    """Run context whose explicit solution file is present but empty."""
    streams = OpenedStreams(
        matrix=cast(MatrixSampleStream, SimpleNamespace(sample_ids=(0,))),
        rhs=None,
        solution=cast(VectorSampleStream, _EmptySolutionStream()),
        parameters=(),
        bindings=(),
    )
    allocation = BindingAllocation(counts=({_STRATEGY: 2},), file_indices=({},))
    return _GenerationRunContext(
        streams=streams,
        allocation=allocation,
        solution_by_position=True,
        cyclic_solution_strategies=frozenset({_STRATEGY}),
    )


def test_empty_solution_file_is_rejected_before_stacking(
    empty_solution_context: _GenerationRunContext,
) -> None:
    """An empty solution file must fail with a clear error, not an empty-stack error."""
    with pytest.raises(ValueError):
        _solution_archive_rows(empty_solution_context, 0, _STRATEGY, 2)
