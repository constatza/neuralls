"""Tests for the remainder-reassignment allocation (pure, no orchestrator)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from neuralls.domain.generation.allocation import archive_units, split_remainder
from neuralls.domain.generation.binding_allocation import _resolve_binding_strategy_counts
from neuralls.domain.generation.bindings import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec
from neuralls.platform.config.models.data_models import GenerationConfig
from tests.fakes.archive_source import InMemoryArchiveSource

GENERATED_STRATEGY = "gaussian_forward"
ARCHIVE_STRATEGY = "solution_archive"

SEED_FOR_REPRODUCIBILITY = 7
SEED_FIRST_DRAW = 0
SEED_SECOND_DRAW = 1


def _multiset(counts: np.ndarray) -> dict[int, int]:
    return dict(Counter(int(c) for c in counts))


@pytest.mark.parametrize(
    ("total", "num_matrices", "expected"),
    [
        (1999, 1000, {2: 999, 1: 1}),
        (7, 3, {3: 1, 2: 2}),
        (3, 5, {1: 3, 0: 2}),
        (1000, 60, {17: 40, 16: 20}),
    ],
    ids=["1999_over_1000", "7_over_3", "3_over_5", "1000_over_60"],
)
def test_split_gives_every_matrix_the_floor_and_r_matrices_one_more(
    total: int, num_matrices: int, expected: dict[int, int]
) -> None:
    """total // n for every matrix, plus one extra unit for each of the total % n matrices."""
    counts = split_remainder(total, num_matrices, seed=SEED_FOR_REPRODUCIBILITY)

    assert counts.shape == (num_matrices,)
    assert _multiset(counts) == expected
    assert int(counts.sum()) == total


def test_split_is_seeded_reproducible() -> None:
    """The same seed gives equal arrays; different seeds pick different extra-unit sets.

    1060 over 60 matrices gives 17 units each with 40 extra units, so the set of
    matrices receiving the extra unit is compared in full, not through one index.
    """
    total, num_matrices, leftover = 1060, 60, 40
    first = split_remainder(total, num_matrices, seed=SEED_FOR_REPRODUCIBILITY)
    second = split_remainder(total, num_matrices, seed=SEED_FOR_REPRODUCIBILITY)
    np.testing.assert_array_equal(first, second)

    base = total // num_matrices
    extra_a = frozenset(
        np.flatnonzero(split_remainder(total, num_matrices, seed=SEED_FIRST_DRAW) > base)
    )
    extra_b = frozenset(
        np.flatnonzero(split_remainder(total, num_matrices, seed=SEED_SECOND_DRAW) > base)
    )
    assert len(extra_a) == len(extra_b) == leftover
    assert extra_a != extra_b


def _matrix_index(pairs: tuple[tuple[int, int], ...]) -> list[int]:
    return [matrix for matrix, _ in pairs]


def _files_by_matrix(pairs: tuple[tuple[int, int], ...]) -> dict[int, list[int]]:
    files: dict[int, list[int]] = {}
    for matrix, file_index in pairs:
        files.setdefault(matrix, []).append(file_index)
    return files


def test_archive_units_hand_example_m3_k4() -> None:
    """t = 0..11 with i = t mod 3 and j = (i + t div 3) mod 4."""
    pairs = archive_units(num_matrices=3, num_files=4, count=12)

    assert pairs == (
        (0, 0), (1, 1), (2, 2),
        (0, 1), (1, 2), (2, 3),
        (0, 2), (1, 3), (2, 0),
        (0, 3), (1, 0), (2, 1),
    )  # fmt: skip


def test_archive_units_m3_k5_n5() -> None:
    """Five units over 3 matrices x 5 files: matrix index [0, 1, 2, 0, 1]; files {0: [0, 1], 1: [1, 2], 2: [2]}."""
    pairs = archive_units(num_matrices=3, num_files=5, count=5)

    assert pairs == ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2))
    assert _matrix_index(pairs) == [0, 1, 2, 0, 1]
    assert _files_by_matrix(pairs) == {0: [0, 1], 1: [1, 2], 2: [2]}


@pytest.mark.parametrize(
    ("num_matrices", "num_files"),
    [(3, 4), (4, 3), (5, 1), (1, 5), (7, 7)],
)
def test_archive_units_full_grid_is_bijection(num_matrices: int, num_files: int) -> None:
    """With count = M * K every (matrix, file) pair appears exactly once."""
    pairs = archive_units(
        num_matrices=num_matrices,
        num_files=num_files,
        count=num_matrices * num_files,
    )

    assert len(pairs) == num_matrices * num_files
    assert len(set(pairs)) == num_matrices * num_files


def test_archive_units_balanced_1999_over_1000_k2() -> None:
    """1999 units over 1000 matrices x 2 files: matrices 0..998 get two, matrix 999 gets one.

    Matrix 998 gets t = 998 (file 0) and t = 1998 (second pass, file 1).
    Matrix 999 gets only t = 999, with file (999 + 0) mod 2 = 1.
    """
    pairs = archive_units(num_matrices=1000, num_files=2, count=1999)
    per_matrix = Counter(matrix for matrix, _ in pairs)
    files = _files_by_matrix(pairs)

    assert all(per_matrix[m] == 2 for m in range(999))
    assert per_matrix[999] == 1
    assert files[998] == [0, 1]
    assert files[999] == [1]


@pytest.mark.parametrize(
    ("num_matrices", "num_files", "count"),
    [(0, 3, 1), (3, 0, 1), (3, 4, -1)],
)
def test_archive_units_rejects_invalid_arguments(
    num_matrices: int, num_files: int, count: int
) -> None:
    """Zero matrices, zero files or a negative count are rejected with ValueError."""
    with pytest.raises(ValueError):
        archive_units(num_matrices=num_matrices, num_files=num_files, count=count)


def test_replacement_true_is_rejected_by_config_validation() -> None:
    """replacement = true fails config validation.

    The config model does not know each strategy's kind, so the rejection covers
    generated and archive strategies alike.
    """
    with pytest.raises(ValidationError):
        GenerationConfig(replacement=True)


@pytest.mark.parametrize(
    ("strategy", "glob"),
    [
        (GENERATED_STRATEGY, None),
        (ARCHIVE_STRATEGY, "/fake/solutions_*.txt"),
    ],
)
def test_replacement_true_is_rejected_by_domain_spec(
    make_bindings: Callable[[int, int], list[SystemBinding]],
    make_mixture: Callable[..., MixtureSpec],
    archive_source: InMemoryArchiveSource,
    strategy: str,
    glob: str | None,
) -> None:
    """DatasetSpec(replacement=True) is refused before any allocation happens."""
    bindings = make_bindings(3, 1)

    with pytest.raises(ValueError):
        _resolve_binding_strategy_counts(
            bindings=bindings,
            spec=DatasetSpec(mixture=make_mixture(strategy, 6, glob=glob), replacement=True),
            num_matrix_samples=3,
            archive_source=archive_source,
        )
