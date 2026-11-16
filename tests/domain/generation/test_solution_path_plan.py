"""An explicit solution_path source with samples = -1 is sized from its row count.

Every binding receives all rows of the solution file, so the plan's total is
(rows in the solution file) x (number of bindings), known before any generation.
The row count is read from the file header or its text lines, never from the values.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.batch_plan import ALL_SAMPLES, plan_batches
from neuralls.domain.generation.binding_allocation import _resolve_binding_strategy_counts
from neuralls.domain.generation.bindings import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec
from neuralls.shared.system_scaling import GershgorinScaling
from tests.fakes.archive_source import InMemoryArchiveSource

_SOLUTION_STRATEGY = "solution_archive"
_NUM_BINDINGS = 2
_NUM_ROWS = 6
_ROW_LENGTH = 3
_SEED = 11


@pytest.fixture
def solution_rows() -> np.ndarray:
    """Seeded solution block with ``_NUM_ROWS`` rows of length ``_ROW_LENGTH``."""
    rng = np.random.default_rng(_SEED)
    return rng.standard_normal((_NUM_ROWS, _ROW_LENGTH))


@pytest.fixture(params=["npy", "txt"])
def solution_file(
    request: pytest.FixtureRequest, archive_source: InMemoryArchiveSource, solution_rows: np.ndarray
) -> tuple[Path, int]:
    """A solution file and its row count: a ``.npy`` block, or a ``.txt`` holding one vector."""
    if request.param == "npy":
        return archive_source.add(Path("archive/solutions.npy"), solution_rows), _NUM_ROWS
    return (
        archive_source.add(Path("archive/solutions.txt"), np.arange(_ROW_LENGTH, dtype=np.float64)),
        1,
    )


def _open_ended_solution_spec() -> DatasetSpec:
    return DatasetSpec(
        mixture=MixtureSpec(counts={_SOLUTION_STRATEGY: ALL_SAMPLES}, seed=0),
        scaling=GershgorinScaling(),
    )


def test_solution_path_total_is_known_before_generation(
    solution_file: tuple[Path, int],
    archive_source: InMemoryArchiveSource,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    path, rows = solution_file
    allocation = _resolve_binding_strategy_counts(
        bindings=make_bindings(_NUM_BINDINGS, 1),
        spec=_open_ended_solution_spec(),
        num_matrix_samples=_NUM_BINDINGS,
        has_solution_source=True,
        solution_rows_total=archive_source.row_count(path),
        archive_source=archive_source,
    )
    plan = plan_batches(allocation)

    assert plan.is_exact
    assert plan.total_rows == _NUM_BINDINGS * rows
