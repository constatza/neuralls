"""Archive strategies never reuse a (matrix, file) pair, for solution and RHS archives."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation import binding_allocation
from neuralls.domain.generation.batch_plan import BindingAllocation
from neuralls.domain.generation.binding_allocation import _resolve_binding_strategy_counts
from neuralls.domain.generation.bindings import SystemBinding
from neuralls.domain.generation.notices import RowCapSource, RowsCapped
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec
from tests.fakes.archive_source import InMemoryArchiveSource

ARCHIVE_DIR = Path("archive")
RHS_ARCHIVE = "rhs_archive"
SOLUTION_ARCHIVE = "solution_archive"
ARCHIVE_STRATEGIES = (SOLUTION_ARCHIVE, RHS_ARCHIVE)
GLOB_OPTION = {SOLUTION_ARCHIVE: "solutions_glob", RHS_ARCHIVE: "rhs_glob"}
FILE_PREFIX = {SOLUTION_ARCHIVE: "solution", RHS_ARCHIVE: "rhs"}


@pytest.fixture(params=ARCHIVE_STRATEGIES)
def strategy(request: pytest.FixtureRequest) -> str:
    """Each archive strategy in turn."""
    return request.param


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive."""
    return InMemoryArchiveSource()


@pytest.fixture
def register_files(archive_source: InMemoryArchiveSource) -> Callable[[str, int], str]:
    """Register ``n`` sorted files for a strategy and return their glob pattern.

    File ``k`` holds the constant vector ``k + 1`` of length 4, so its position in
    the sorted pool is recoverable from its contents.
    """

    def _register(strategy: str, n_files: int) -> str:
        prefix = FILE_PREFIX[strategy]
        for idx in range(n_files):
            archive_source.add(ARCHIVE_DIR / f"{prefix}_{idx:03d}.txt", np.full(4, float(idx + 1)))
        return str(ARCHIVE_DIR / f"{prefix}_*.txt")

    return _register


@pytest.fixture
def matrix_bindings() -> list[SystemBinding]:
    """Three matrices, one binding each."""
    return [SystemBinding(sample_id=i, matrix_sample_id=i) for i in range(3)]


@pytest.fixture
def one_matrix_two_bindings() -> list[SystemBinding]:
    """One matrix shared by two bindings (two right-hand sides on the same system)."""
    return [
        SystemBinding(sample_id=0, matrix_sample_id=0),
        SystemBinding(sample_id=1, matrix_sample_id=0),
    ]


def _spec(strategy: str, glob_pattern: str, count: int) -> DatasetSpec:
    """Dataset spec with one archive strategy over the given glob."""
    glob_key = GLOB_OPTION[strategy]
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={strategy: count},
            seed=0,
            strategy_overrides={strategy: {glob_key: glob_pattern}},
        ),
        replacement=False,
    )


def _files_by_binding(allocation: BindingAllocation, strategy: str) -> list[list[int]]:
    """Explicit archive file indices per binding (empty when a binding has none)."""
    return [list(indices.get(strategy, ())) for indices in allocation.file_indices]


def _pairs(allocation: BindingAllocation, strategy: str) -> list[tuple[int, int]]:
    """(matrix/binding, file) pairs; one binding per matrix in these fixtures."""
    return [
        (binding_idx, file_idx)
        for binding_idx, files in enumerate(_files_by_binding(allocation, strategy))
        for file_idx in files
    ]


def _counts(allocation: BindingAllocation, strategy: str) -> list[int]:
    return [counts[strategy] for counts in allocation.counts]


def test_cyclic_assignment_over_the_archive_glob(
    strategy: str,
    register_files: Callable[[str, int], str],
    matrix_bindings: list[SystemBinding],
    archive_source: InMemoryArchiveSource,
) -> None:
    """A count of 5 over 5 files uses one cyclic (matrix, file) map for every archive strategy."""
    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=_spec(strategy, register_files(strategy, 5), 5),
        num_matrix_samples=3,
        archive_source=archive_source,
    )

    assert _counts(allocation, strategy) == [2, 2, 1]
    assert _files_by_binding(allocation, strategy) == [[0, 1], [1, 2], [2]]
    pairs = _pairs(allocation, strategy)
    assert len(set(pairs)) == len(pairs) == 5


def test_all_samples_emits_every_pair_once(
    strategy: str,
    register_files: Callable[[str, int], str],
    matrix_bindings: list[SystemBinding],
    archive_source: InMemoryArchiveSource,
) -> None:
    """samples = -1 emits all M*K pairs, each once, cyclically."""
    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=_spec(strategy, register_files(strategy, 5), -1),
        num_matrix_samples=3,
        archive_source=archive_source,
    )

    assert _counts(allocation, strategy) == [5, 5, 5]
    assert _files_by_binding(allocation, strategy) == [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
    ]
    pairs = _pairs(allocation, strategy)
    assert len(set(pairs)) == 15
    assert [binding for binding, _ in pairs] == [0] * 5 + [1] * 5 + [2] * 5


def test_request_above_m_times_k_is_capped_with_one_notice(
    strategy: str,
    register_files: Callable[[str, int], str],
    matrix_bindings: list[SystemBinding],
    archive_source: InMemoryArchiveSource,
) -> None:
    """A count above M*K is capped at M*K and reported once, naming the strategy."""
    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=_spec(strategy, register_files(strategy, 5), 20),
        num_matrix_samples=3,
        archive_source=archive_source,
    )

    assert _counts(allocation, strategy) == [5, 5, 5]
    assert len(set(_pairs(allocation, strategy))) == 15
    assert allocation.notices == (RowsCapped(strategy, 20, 15, RowCapSource.ARCHIVE_PAIRS),)


def test_single_matrix_several_bindings_rejected_before_glob_read(
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
    one_matrix_two_bindings: list[SystemBinding],
    archive_source: InMemoryArchiveSource,
) -> None:
    """One matrix with several archive bindings is rejected before any glob read."""

    def _glob_must_not_be_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("archive glob read before the repeat check")

    monkeypatch.setattr(binding_allocation, "select_archive_files", _glob_must_not_be_read)
    with pytest.raises(ValueError):
        _resolve_binding_strategy_counts(
            bindings=one_matrix_two_bindings,
            spec=_spec(strategy, f"/fake/{FILE_PREFIX[strategy]}_*.txt", 3),
            num_matrix_samples=1,
            archive_source=archive_source,
        )


def test_solution_source_and_archive_glob_are_rejected_together(
    strategy: str,
    register_files: Callable[[str, int], str],
    matrix_bindings: list[SystemBinding],
    archive_source: InMemoryArchiveSource,
) -> None:
    """A per-binding solution_path and an archive glob are both sources: reject the mix.

    Passing the glob through unchanged would let the archive draw files outside the
    (matrix, file) assignment, so the combination is an explicit error.
    """
    with pytest.raises(ValueError):
        _resolve_binding_strategy_counts(
            bindings=matrix_bindings,
            spec=_spec(strategy, register_files(strategy, 5), 3),
            num_matrix_samples=3,
            has_solution_source=True,
            archive_source=archive_source,
        )


@pytest.mark.parametrize("count", [1, 4, 7, 12, -1])
def test_archive_pairs_never_repeat_across_matrices(
    strategy: str,
    count: int,
    register_files: Callable[[str, int], str],
    matrix_bindings: list[SystemBinding],
    archive_source: InMemoryArchiveSource,
) -> None:
    """Every (matrix, file) pair is distinct for each archive strategy and several counts."""
    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=_spec(strategy, register_files(strategy, 4), count),
        num_matrix_samples=3,
        archive_source=archive_source,
    )

    pairs = _pairs(allocation, strategy)
    assert len(set(pairs)) == len(pairs)


def test_both_archives_in_one_build_never_repeat_pairs(
    register_files: Callable[[str, int], str],
    matrix_bindings: list[SystemBinding],
    archive_source: InMemoryArchiveSource,
) -> None:
    """solution_archive and rhs_archive together over several matrices: no repeated pair.

    Checks the full allocation a multi-matrix build uses, for both strategies at once,
    so each strategy's (matrix, file) pairs are collected from the same resolution.
    """
    spec = DatasetSpec(
        mixture=MixtureSpec(
            counts={SOLUTION_ARCHIVE: 7, RHS_ARCHIVE: 7},
            seed=0,
            strategy_overrides={
                strategy: {GLOB_OPTION[strategy]: register_files(strategy, 4)}
                for strategy in ARCHIVE_STRATEGIES
            },
        ),
        replacement=False,
    )

    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=spec,
        num_matrix_samples=3,
        archive_source=archive_source,
    )

    for strategy in ARCHIVE_STRATEGIES:
        pairs = _pairs(allocation, strategy)
        assert len(pairs) == sum(_counts(allocation, strategy))
        assert len(set(pairs)) == len(pairs)
