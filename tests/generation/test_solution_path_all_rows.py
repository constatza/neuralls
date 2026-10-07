"""Explicit ``solution_path`` files: ``samples=-1`` gives every binding every row."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.helpers import solution_row_count
from neuralls.domain.generation.source_streams import open_vector_stream
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import load_dense_training_arrays

N_UNKNOWNS = 4
N_SOLUTION_ROWS = 4
SEED = 7


@pytest.fixture
def two_spd_matrices(tmp_path: Path) -> Path:
    """Write two SPD matrices ``A_0.txt``/``A_1.txt`` and return their directory."""
    rng = np.random.default_rng(SEED)
    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir()
    for i in range(2):
        base = rng.standard_normal((N_UNKNOWNS, N_UNKNOWNS))
        matrix = base @ base.T + N_UNKNOWNS * np.eye(N_UNKNOWNS)
        np.savetxt(mat_dir / f"A_{i}.txt", matrix)
    return mat_dir


@pytest.fixture
def solution_rows() -> np.ndarray:
    """Seeded ``(N_SOLUTION_ROWS, N_UNKNOWNS)`` block of distinct solution vectors."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal((N_SOLUTION_ROWS, N_UNKNOWNS))


@pytest.fixture
def solution_npy(tmp_path: Path, solution_rows: np.ndarray) -> Path:
    """Persist the solution block as a stacked ``.npy`` file."""
    path = tmp_path / "solutions.npy"
    np.save(path, solution_rows)
    return path


@pytest.fixture
def solution_txt(tmp_path: Path) -> Path:
    """Persist a ``.txt`` solution file with four lines of values."""
    rng = np.random.default_rng(SEED)
    path = tmp_path / "solutions.txt"
    np.savetxt(path, rng.standard_normal(N_UNKNOWNS))
    return path


@pytest.fixture
def build_archive_dataset(
    two_spd_matrices: Path,
    tmp_path: Path,
) -> Callable[[Path, int], tuple[np.ndarray, np.ndarray]]:
    """Build a ``solution_archive`` dataset from an explicit solution file.

    Returns a callable taking ``(solution_file, samples)`` and returning the stored
    ``(rhs, solutions)`` arrays.
    """

    def _build(solution_file: Path, samples: int) -> tuple[np.ndarray, np.ndarray]:
        dataset_dir = str(tmp_path / f"dataset_{samples}")
        build_dataset(
            SourceSpec(
                matrix_path=str(two_spd_matrices / "A_*.txt"),
                solution_path=str(solution_file),
                sample_id_regex=r"(\d+)(?!.*\d)",
            ),
            DatasetSpec(mixture=MixtureSpec(counts={"solution_archive": samples})),
            dataset_dir,
        )
        return load_dense_training_arrays(dataset_dir)

    return _build


def test_all_samples_with_explicit_solution_file_emits_every_row_per_binding(
    build_archive_dataset: Callable[[Path, int], tuple[np.ndarray, np.ndarray]],
    solution_npy: Path,
    solution_rows: np.ndarray,
) -> None:
    """``samples=-1`` on an explicit file: each of the two bindings emits all four rows in file order."""
    _, solutions = build_archive_dataset(solution_npy, -1)

    assert solutions.shape == (2 * N_SOLUTION_ROWS, N_UNKNOWNS)
    np.testing.assert_allclose(solutions[:N_SOLUTION_ROWS], solution_rows)
    np.testing.assert_allclose(solutions[N_SOLUTION_ROWS:], solution_rows)


def test_explicit_count_draws_cyclic_rows_per_binding(
    build_archive_dataset: Callable[[Path, int], tuple[np.ndarray, np.ndarray]],
    solution_npy: Path,
    solution_rows: np.ndarray,
) -> None:
    """``samples=2`` on an explicit file: binding ``b`` draws rows ``(b + p) mod K``."""
    _, solutions = build_archive_dataset(solution_npy, 2)

    assert solutions.shape == (2 * 2, N_UNKNOWNS)
    np.testing.assert_allclose(solutions[0:2], solution_rows[[0, 1]])
    np.testing.assert_allclose(solutions[2:4], solution_rows[[1, 2]])


def test_explicit_count_above_file_rows_is_capped_with_one_warning(
    build_archive_dataset: Callable[[Path, int], tuple[np.ndarray, np.ndarray]],
    solution_npy: Path,
    solution_rows: np.ndarray,
    warning_messages: list[str],
) -> None:
    """``samples=6`` on a 4-row file: each binding emits 4 rows, reported in one warning."""
    _, solutions = build_archive_dataset(solution_npy, 6)

    assert solutions.shape == (2 * N_SOLUTION_ROWS, N_UNKNOWNS)
    np.testing.assert_allclose(solutions[0:4], solution_rows[[0, 1, 2, 3]])
    np.testing.assert_allclose(solutions[4:8], solution_rows[[1, 2, 3, 0]])
    capped = [message for message in warning_messages if "solution" in message and "6" in message]
    assert len(capped) == 1
    assert str(N_SOLUTION_ROWS) in capped[0]


def test_explicit_count_equal_to_file_rows_draws_every_row_in_cyclic_order(
    build_archive_dataset: Callable[[Path, int], tuple[np.ndarray, np.ndarray]],
    solution_npy: Path,
    solution_rows: np.ndarray,
) -> None:
    """``samples=K`` on a K-row file: binding ``b`` emits rows ``(b, b+1, b+2, b+3) mod K``."""
    _, solutions = build_archive_dataset(solution_npy, N_SOLUTION_ROWS)

    np.testing.assert_allclose(solutions[0:4], solution_rows[[0, 1, 2, 3]])
    np.testing.assert_allclose(solutions[4:8], solution_rows[[1, 2, 3, 0]])


def test_text_solution_file_row_semantics_match_reader(solution_txt: Path) -> None:
    """The row count of a ``.txt`` solution file equals the rows the vector reader loads."""
    reader_rows = len(open_vector_stream(str(solution_txt)).sample_ids)

    assert solution_row_count(solution_txt) == reader_rows
