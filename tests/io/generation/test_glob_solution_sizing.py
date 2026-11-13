"""A glob solution_path is sized from its matched file list, like a glob archive.

The plan's row total must equal the rows the streamed build writes, for every count rule:
``samples=-1`` takes the full pool on each binding, an explicit count is drawn cyclically
from the pool, and a count above the pool is capped at it with one warning.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from tests.dataset_support import build_test_dataset

from neuralls.domain.generation.batch_plan import BindingAllocation, plan_batches
from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.notices import RowCapSource, RowsCapped
from neuralls.domain.generation.orchestration import _prepare_generation_context
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.archive_source import FsArchiveSource
from neuralls.platform.storage.datasets import load_dense_training_arrays
from neuralls.platform.storage.matrix_readers import read_dense_npy, read_matrix

ARCHIVE_SOURCE = FsArchiveSource()


_READER = partial(read_matrix, lazy=True)
_READERS = MatrixReaders(generic=_READER, dense=partial(read_dense_npy, lazy=True))

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


def _allocation(source: SourceSpec, spec: DatasetSpec) -> BindingAllocation:
    return _prepare_generation_context(
        source, spec, readers=_READERS, archive_source=ARCHIVE_SOURCE
    ).allocation


def _planned_total(source: SourceSpec, spec: DatasetSpec) -> int:
    plan = plan_batches(_allocation(source, spec))
    assert plan.is_exact
    return plan.total_rows


def _written_rows(source: SourceSpec, spec: DatasetSpec, dataset_dir: Path) -> int:
    build_test_dataset(source, spec, str(dataset_dir))
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


def test_glob_solution_path_count_above_pool_is_capped_with_one_notice(
    matrix_dir: Path,
    solution_dir_factory: Callable[[int], Path],
    tmp_path: Path,
) -> None:
    """An explicit count above K=2 is capped at the pool, reported once, and sized to match."""
    num_files = 2
    solution_dir = solution_dir_factory(num_files)
    source = _source(matrix_dir, str(solution_dir / "x_*.txt"))
    spec = _spec(5)

    assert _planned_total(source, spec) == _NUM_MATRICES * num_files
    assert _allocation(source, spec).notices == (
        RowsCapped("solution_archive", 5, num_files, RowCapSource.SOLUTION_FILE),
    )
    assert _written_rows(source, spec, tmp_path / "dataset") == _NUM_MATRICES * num_files
