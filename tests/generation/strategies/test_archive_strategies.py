"""Tests for the rhs_archive and solution_archive strategies (one table, one test per rule)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.interfaces import SolvedRows
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.generation_support import solved_rows

ARCHIVE_DIR = Path("archive")
FILE_COUNT = 5


@dataclass(frozen=True, slots=True)
class ArchiveKind:
    """What differs between the two archive strategies.

    Attributes:
        strategy: Registered strategy name.
        glob_option: Config key holding the archive glob.
        prefix: Filename prefix of the archive files.
        seed: Seed of the random vectors the archive holds.
        loaded: The rows read from the archive (the other array is derived from them).
    """

    strategy: str
    glob_option: str
    prefix: str
    seed: int
    loaded: Callable[[SolvedRows], np.ndarray | None]


RHS_ARCHIVE = ArchiveKind("rhs_archive", "rhs_glob", "rhs", 3, lambda rows: rows.rhs)
SOLUTION_ARCHIVE = ArchiveKind(
    "solution_archive", "solutions_glob", "sol", 5, lambda rows: rows.solutions
)


@pytest.fixture(params=[RHS_ARCHIVE, SOLUTION_ARCHIVE], ids=lambda kind: kind.strategy)
def kind(request: pytest.FixtureRequest) -> ArchiveKind:
    """Each archive strategy in turn."""
    return request.param


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive."""
    return InMemoryArchiveSource()


@pytest.fixture
def register_archive(
    archive_source: InMemoryArchiveSource, spd_matrix: np.ndarray
) -> Callable[[ArchiveKind], tuple[list[Path], str]]:
    """Register five archive files for a kind in memory; return them and their glob."""

    def _register(kind: ArchiveKind) -> tuple[list[Path], str]:
        rng = np.random.default_rng(kind.seed)
        files = [
            archive_source.add(
                ARCHIVE_DIR / f"{kind.prefix}_{i:03d}.txt",
                rng.standard_normal(spd_matrix.shape[0]),
            )
            for i in range(FILE_COUNT)
        ]
        return files, str(ARCHIVE_DIR / f"{kind.prefix}_*.txt")

    return _register


def _run(
    kind: ArchiveKind,
    matrix: np.ndarray,
    archive_source: InMemoryArchiveSource,
    **cfg: object,
) -> SolvedRows:
    return solved_rows(
        run_generation(kind.strategy, matrix, cfg=cfg, archive_source=archive_source)
    )


def test_archive_rows_have_the_requested_shape(
    kind: ArchiveKind,
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
) -> None:
    n = spd_matrix.shape[0]
    _files, glob_pattern = register_archive(kind)

    result = _run(kind, spd_matrix, archive_source, **{kind.glob_option: glob_pattern}, samples=3)

    assert result.rhs is not None
    assert result.solutions is not None
    assert result.rhs.shape == (3, n)
    assert result.solutions.shape == (3, n)


def test_all_samples_loads_every_file(
    kind: ArchiveKind,
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
) -> None:
    _files, glob_pattern = register_archive(kind)

    result = _run(kind, spd_matrix, archive_source, **{kind.glob_option: glob_pattern}, samples=-1)

    loaded = kind.loaded(result)
    assert loaded is not None
    assert loaded.shape[0] == FILE_COUNT


def test_shuffle_with_the_same_seed_is_repeatable(
    kind: ArchiveKind,
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
) -> None:
    _files, glob_pattern = register_archive(kind)
    cfg = {kind.glob_option: glob_pattern, "samples": 3, "shuffle": True, "seed": 42}

    first = kind.loaded(_run(kind, spd_matrix, archive_source, **cfg))
    second = kind.loaded(_run(kind, spd_matrix, archive_source, **cfg))

    assert first is not None
    assert second is not None
    np.testing.assert_array_equal(first, second)


def test_missing_glob_option_is_rejected(
    kind: ArchiveKind, archive_source: InMemoryArchiveSource
) -> None:
    with pytest.raises(ValueError):
        _run(kind, np.eye(5), archive_source, samples=1)


def test_more_samples_than_files_is_rejected(
    kind: ArchiveKind,
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
) -> None:
    _files, glob_pattern = register_archive(kind)

    with pytest.raises(ValueError):
        _run(kind, spd_matrix, archive_source, **{kind.glob_option: glob_pattern}, samples=100)


def test_rhs_archive_solutions_solve_the_system(
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
) -> None:
    """Returned solutions satisfy A @ x = b within the CG tolerance."""
    _files, glob_pattern = register_archive(RHS_ARCHIVE)
    result = _run(
        RHS_ARCHIVE,
        spd_matrix,
        archive_source,
        rhs_glob=glob_pattern,
        samples=2,
        solve_config={"method": "cg", "rtol": 1e-10, "max_iters": 100},
    )

    assert result.rhs is not None
    assert result.solutions is not None
    relative = np.linalg.norm(result.rhs - result.solutions @ spd_matrix.T, axis=1) / (
        np.linalg.norm(result.rhs, axis=1)
    )
    assert np.all(relative < 1e-6)


def test_solution_archive_rhs_is_a_times_solution(
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
) -> None:
    _files, glob_pattern = register_archive(SOLUTION_ARCHIVE)
    result = _run(
        SOLUTION_ARCHIVE, spd_matrix, archive_source, solutions_glob=glob_pattern, samples=3
    )

    assert result.rhs is not None
    assert result.solutions is not None
    np.testing.assert_allclose(result.rhs, result.solutions @ spd_matrix.T, rtol=1e-12)


@pytest.mark.parametrize("shuffle", [False, True], ids=["in-order", "shuffled"])
def test_solution_archive_skip_offsets_the_selection(
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
    shuffle: bool,
) -> None:
    """skip drops the first files of the (shuffled) selection before loading vectors."""
    files, glob_pattern = register_archive(SOLUTION_ARCHIVE)
    seed, skip, count = 42, 2, 2

    result = _run(
        SOLUTION_ARCHIVE,
        spd_matrix,
        archive_source,
        solutions_glob=glob_pattern,
        samples=count,
        shuffle=shuffle,
        seed=seed,
        skip=skip,
    )

    order = (
        np.random.default_rng(seed).permutation(len(files)) if shuffle else np.arange(len(files))
    )
    expected = np.array(
        [archive_source.load_vector(files[idx]) for idx in order[skip : skip + count]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(result.solutions, expected)


def test_solution_archive_skip_plus_samples_beyond_the_files_is_rejected(
    spd_matrix: np.ndarray,
    register_archive: Callable[[ArchiveKind], tuple[list[Path], str]],
    archive_source: InMemoryArchiveSource,
) -> None:
    _files, glob_pattern = register_archive(SOLUTION_ARCHIVE)

    with pytest.raises(ValueError):
        _run(
            SOLUTION_ARCHIVE,
            spd_matrix,
            archive_source,
            solutions_glob=glob_pattern,
            samples=2,
            skip=4,
        )
