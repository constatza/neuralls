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
from neuralls.domain.generation.helpers import solution_row_count
from neuralls.domain.generation.orchestration import _resolve_binding_strategy_counts
from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec

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


@pytest.fixture
def solution_npy(tmp_path: Path, solution_rows: np.ndarray) -> Path:
    path = tmp_path / "solutions.npy"
    np.save(path, solution_rows)
    return path


@pytest.fixture
def solution_txt(tmp_path: Path) -> Path:
    """A .txt solution file holding one vector: the vector reader loads it as one sample."""
    path = tmp_path / "solutions.txt"
    np.savetxt(path, np.arange(_ROW_LENGTH, dtype=np.float64))
    return path


@pytest.fixture
def unsized_solution(tmp_path: Path) -> Path:
    """A solution file in a format whose row count is not read cheaply."""
    path = tmp_path / "solutions.csv"
    path.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")
    return path


def _open_ended_solution_spec() -> DatasetSpec:
    return DatasetSpec(
        mixture=MixtureSpec(counts={_SOLUTION_STRATEGY: ALL_SAMPLES}, seed=0),
        normalize="matrix",
    )


def test_solution_path_total_is_known_before_generation(
    solution_npy: Path,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    spec = _open_ended_solution_spec()
    bindings = make_bindings(_NUM_BINDINGS, 1)

    allocation = _resolve_binding_strategy_counts(
        bindings=bindings,
        spec=spec,
        num_matrix_samples=_NUM_BINDINGS,
        has_solution_source=True,
        solution_rows_total=solution_row_count(solution_npy),
    )
    plan = plan_batches(allocation)

    assert plan.is_exact
    assert plan.total_rows == _NUM_BINDINGS * _NUM_ROWS


def test_solution_path_text_file_row_count(
    solution_txt: Path,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    spec = _open_ended_solution_spec()
    bindings = make_bindings(_NUM_BINDINGS, 1)

    allocation = _resolve_binding_strategy_counts(
        bindings=bindings,
        spec=spec,
        num_matrix_samples=_NUM_BINDINGS,
        has_solution_source=True,
        solution_rows_total=solution_row_count(solution_txt),
    )
    plan = plan_batches(allocation)

    assert plan.is_exact
    assert plan.total_rows == _NUM_BINDINGS * 1


def test_solution_path_row_count_failure_is_named(
    unsized_solution: Path,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    spec = _open_ended_solution_spec()
    bindings = make_bindings(_NUM_BINDINGS, 1)

    with pytest.raises(ValueError, match=r"solutions\.csv.*\.csv"):
        _resolve_binding_strategy_counts(
            bindings=bindings,
            spec=spec,
            num_matrix_samples=_NUM_BINDINGS,
            has_solution_source=True,
            solution_rows_total=solution_row_count(unsized_solution),
        )
