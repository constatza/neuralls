"""Tests for compute_dataset_fingerprint's determinism and change-sensitivity.

This fingerprint is the sole signal the training/comparison reuse-check
cascade uses to detect "the underlying dataset changed" — these tests pin
down exactly what it does and does not notice.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from neuralls.platform.caching import compute_dataset_fingerprint


@pytest.fixture
def dataset_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Three small files standing in for a dataset's matrix/rhs/solutions artifacts."""
    matrix = tmp_path / "matrix.npy"
    rhs = tmp_path / "rhs.npy"
    solutions = tmp_path / "solutions.npy"
    matrix.write_bytes(b"matrix-bytes")
    rhs.write_bytes(b"rhs-bytes")
    solutions.write_bytes(b"solutions-bytes")
    return matrix, rhs, solutions


def _bump_mtime(path: Path, *, seconds: float = 1.0) -> None:
    """Advance a file's mtime deterministically, without depending on wall-clock timing."""
    stat = path.stat()
    new_time = stat.st_mtime + seconds
    os.utime(path, (new_time, new_time))


def test_same_files_produce_the_same_fingerprint_every_time(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    """Repeated calls against unchanged files are fully deterministic."""
    first = compute_dataset_fingerprint(dataset_files)
    second = compute_dataset_fingerprint(dataset_files)
    assert first == second


def test_fingerprint_is_independent_of_path_order(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    """Callers pass artifact paths in whatever order they're resolved in — the
    fingerprint must not depend on that order, or two equivalent generations
    could spuriously look different."""
    matrix, rhs, solutions = dataset_files
    forward = compute_dataset_fingerprint((matrix, rhs, solutions))
    reversed_order = compute_dataset_fingerprint((solutions, rhs, matrix))
    assert forward == reversed_order


def test_regenerating_a_file_changes_the_fingerprint(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    """Rewriting one file (as a real dataset regeneration would) changes its
    mtime, which must change the overall fingerprint — this is the exact
    mechanism the training reuse-check relies on to notice a stale dataset."""
    matrix, rhs, solutions = dataset_files
    before = compute_dataset_fingerprint((matrix, rhs, solutions))

    _bump_mtime(rhs)

    after = compute_dataset_fingerprint((matrix, rhs, solutions))
    assert before != after


def test_different_file_size_changes_the_fingerprint(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    matrix, rhs, solutions = dataset_files
    before = compute_dataset_fingerprint((matrix, rhs, solutions))

    solutions.write_bytes(b"a much longer solutions payload than before")
    _bump_mtime(solutions)

    after = compute_dataset_fingerprint((matrix, rhs, solutions))
    assert before != after


def test_adding_or_removing_a_path_changes_the_fingerprint(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    matrix, rhs, solutions = dataset_files
    two_paths = compute_dataset_fingerprint((matrix, rhs))
    three_paths = compute_dataset_fingerprint((matrix, rhs, solutions))
    assert two_paths != three_paths


def test_unrelated_untouched_files_do_not_affect_the_fingerprint(
    tmp_path: Path, dataset_files: tuple[Path, Path, Path]
) -> None:
    """Only the given paths are fingerprinted — an unrelated file appearing in
    the same directory must not perturb the result."""
    matrix, rhs, solutions = dataset_files
    before = compute_dataset_fingerprint((matrix, rhs, solutions))

    (tmp_path / "unrelated.txt").write_text("noise", encoding="utf-8")

    after = compute_dataset_fingerprint((matrix, rhs, solutions))
    assert before == after
