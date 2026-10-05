"""Tests for the remainder-reassignment allocation (pure, no orchestrator)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from neuralls.domain.generation.allocation import archive_units, split_remainder
from neuralls.domain.generation.orchestration import _resolve_binding_strategy_counts
from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec
from neuralls.platform.config.models.data_models import GenerationConfig

GENERATED_STRATEGY = "gaussian_forward"
ARCHIVE_STRATEGY = "solution_archive"

SEED_FOR_REPRODUCIBILITY = 7
SEED_FIRST_DRAW = 0
SEED_SECOND_DRAW = 1


def _multiset(counts: np.ndarray) -> dict[int, int]:
    return dict(Counter(int(c) for c in counts))


@pytest.mark.parametrize(
    ("total", "num_matrices"),
    [(1999, 1000), (7, 3), (1000, 60)],
)
def test_split_is_at_most_one_unit_apart_and_sums_to_total(total: int, num_matrices: int) -> None:
    """No matrix gets more than one remainder unit: max - min <= 1, sum == total."""
    counts = split_remainder(total, num_matrices, seed=SEED_FOR_REPRODUCIBILITY)

    assert counts.shape == (num_matrices,)
    assert int(counts.max()) - int(counts.min()) <= 1
    assert int(counts.sum()) == total


def test_split_1999_over_1000_gives_999_twos_and_one_one() -> None:
    """1999 // 1000 = 1, r = 999: 999 matrices at 2, one at 1."""
    counts = split_remainder(1999, 1000, seed=SEED_FOR_REPRODUCIBILITY)

    assert _multiset(counts) == {2: 999, 1: 1}


def test_split_7_over_3_is_3_2_2() -> None:
    """7 // 3 = 2, r = 1: one matrix at 3, two at 2."""
    counts = split_remainder(7, 3, seed=SEED_FOR_REPRODUCIBILITY)

    assert _multiset(counts) == {3: 1, 2: 2}


def test_split_3_over_5_gives_three_ones_and_two_zeros() -> None:
    """3 // 5 = 0, r = 3: three matrices at 1, two at 0."""
    counts = split_remainder(3, 5, seed=SEED_FOR_REPRODUCIBILITY)

    assert _multiset(counts) == {1: 3, 0: 2}


def test_split_1000_over_60_gives_40_at_17_and_20_at_16() -> None:
    """1000 // 60 = 16, r = 40: forty matrices at 17, twenty at 16."""
    counts = split_remainder(1000, 60, seed=SEED_FOR_REPRODUCIBILITY)

    assert _multiset(counts) == {17: 40, 16: 20}


def test_split_is_seeded_reproducible() -> None:
    """The same seed gives equal arrays; different seeds pick different single-unit matrices."""
    first = split_remainder(1999, 1000, seed=SEED_FOR_REPRODUCIBILITY)
    second = split_remainder(1999, 1000, seed=SEED_FOR_REPRODUCIBILITY)
    np.testing.assert_array_equal(first, second)

    draw_a = split_remainder(1999, 1000, seed=SEED_FIRST_DRAW)
    draw_b = split_remainder(1999, 1000, seed=SEED_SECOND_DRAW)
    assert int(np.argmin(draw_a)) != int(np.argmin(draw_b))


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
    with pytest.raises(ValidationError, match="replacement = true is not supported"):
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
    strategy: str,
    glob: str | None,
) -> None:
    """DatasetSpec(replacement=True) is refused before any allocation happens."""
    bindings = make_bindings(3, 1)

    with pytest.raises(ValueError, match="replacement = true is not supported"):
        _resolve_binding_strategy_counts(
            bindings=bindings,
            spec=DatasetSpec(mixture=make_mixture(strategy, 6, glob=glob), replacement=True),
            num_matrix_samples=3,
        )
