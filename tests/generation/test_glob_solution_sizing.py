"""A glob solution_path is sized from its matched file list, like a glob archive.

The plan's row total must equal the rows the streamed build writes, for every count rule:
``samples=-1`` takes the full pool on each binding, an explicit count is drawn cyclically
from the pool, and a count above the pool is capped at it with one warning.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.batch_plan import plan_batches
from neuralls.domain.generation.orchestration import _prepare_generation_context
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import load_dense_training_arrays

_NUM_MATRICES = 2
_N = 4
_SEED = 99


@pytest.fixture
def matrix_dir(tmp_path: Path) -> Path:
    """Two seeded 4x4 SPD matrices named by integer id."""
    rng = np.random.default_rng(seed=_SEED)
    directory = tmp_path / "matrices"
    directory.mkdir()
    for i in range(_NUM_MATRICES):
        a = rng.standard_normal((_N, _N))
        np.savetxt(directory / f"A_{i}.txt", a @ a.T + _N * np.eye(_N))
    return directory


@pytest.fixture
def solution_dir_factory(tmp_path: Path) -> Callable[[int], Path]:
    """Build a directory of K seeded solution files named by integer id, K from the test."""

    def _make(num_files: int) -> Path:
        rng = np.random.default_rng(seed=_SEED + num_files)
        directory = tmp_path / f"solutions_{num_files}"
        directory.mkdir()
        for i in range(num_files):
            np.savetxt(directory / f"x_{i}.txt", rng.standard_normal(_N))
        return directory

    return _make


def _source(matrix_dir: Path, solution_glob: str) -> SourceSpec:
    return SourceSpec(
        matrix_path=str(matrix_dir / "A_*.txt"),
        solution_path=solution_glob,
        sample_id_regex=r"(\d+)(?!.*\d)",
    )


def _spec(count: int) -> DatasetSpec:
    return DatasetSpec(mixture=MixtureSpec(counts={"solution_archive": count}))


def _planned_total(source: SourceSpec, spec: DatasetSpec) -> int:
    context = _prepare_generation_context(source, spec)
    plan = plan_batches(context.allocation)
    assert plan.is_exact
    return plan.total_rows


def _written_rows(source: SourceSpec, spec: DatasetSpec, dataset_dir: Path) -> int:
    build_dataset(source, spec, str(dataset_dir))
    rhs, solutions = load_dense_training_arrays(str(dataset_dir))
    assert rhs.shape == solutions.shape
    return rhs.shape[0]


def test_glob_solution_path_all_samples_is_sized_and_exact(
    matrix_dir: Path, solution_dir_factory: Callable[[int], Path], tmp_path: Path
) -> None:
    """samples=-1 on K matched files gives every binding the full pool: M x K rows, exact."""
    num_files = 3
    solution_dir = solution_dir_factory(num_files)
    source = _source(matrix_dir, str(solution_dir / "x_*.txt"))
    spec = _spec(-1)

    assert _planned_total(source, spec) == _NUM_MATRICES * num_files
    assert _written_rows(source, spec, tmp_path / "dataset") == _NUM_MATRICES * num_files


def test_glob_solution_path_explicit_count_matches_the_written_rows(
    matrix_dir: Path, solution_dir_factory: Callable[[int], Path], tmp_path: Path
) -> None:
    """Explicit count 1 per binding over K=3 files: the plan total equals the written rows."""
    solution_dir = solution_dir_factory(3)
    source = _source(matrix_dir, str(solution_dir / "x_*.txt"))
    spec = _spec(1)

    assert _planned_total(source, spec) == _NUM_MATRICES
    assert _written_rows(source, spec, tmp_path / "dataset") == _NUM_MATRICES


def test_glob_solution_path_count_above_pool_is_capped_with_one_warning(
    matrix_dir: Path,
    solution_dir_factory: Callable[[int], Path],
    tmp_path: Path,
    warning_messages: list[str],
) -> None:
    """An explicit count above K=2 is capped at the pool, warned once, and sized to match."""
    num_files = 2
    solution_dir = solution_dir_factory(num_files)
    source = _source(matrix_dir, str(solution_dir / "x_*.txt"))
    spec = _spec(5)

    assert _planned_total(source, spec) == _NUM_MATRICES * num_files
    capped = [m for m in warning_messages if "solution file" in m]
    assert len(capped) == 1
    assert _written_rows(source, spec, tmp_path / "dataset") == _NUM_MATRICES * num_files
