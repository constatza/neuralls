"""Archive strategies never reuse a (matrix, file) pair, for solution and RHS archives."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.generation import orchestration
from neuralls.domain.generation.orchestration import (
    BindingAllocation,
    _resolve_binding_strategy_counts,
)
from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec

RHS_ARCHIVE = "rhs_archive"
SOLUTION_ARCHIVE = "solution_archive"
ARCHIVE_STRATEGIES = (SOLUTION_ARCHIVE, RHS_ARCHIVE)


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
    glob_key = "rhs_glob" if strategy == RHS_ARCHIVE else "solutions_glob"
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


def test_rhs_archive_cyclic_assignment_over_rhs_glob(
    write_rhs_files: Callable[[int], str],
    matrix_bindings: list[SystemBinding],
) -> None:
    """rhs_archive with count 5 over 5 RHS files uses the same cyclic map as solutions."""
    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=_spec(RHS_ARCHIVE, write_rhs_files(5), 5),
        num_matrix_samples=3,
    )

    assert _counts(allocation, RHS_ARCHIVE) == [2, 2, 1]
    assert _files_by_binding(allocation, RHS_ARCHIVE) == [[0, 1], [1, 2], [2]]
    pairs = _pairs(allocation, RHS_ARCHIVE)
    assert len(set(pairs)) == len(pairs) == 5


def test_rhs_archive_all_samples_emits_every_pair_once(
    write_rhs_files: Callable[[int], str],
    matrix_bindings: list[SystemBinding],
) -> None:
    """samples = -1 on rhs_archive emits all M*K pairs, each once, cyclically."""
    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=_spec(RHS_ARCHIVE, write_rhs_files(5), -1),
        num_matrix_samples=3,
    )

    assert _counts(allocation, RHS_ARCHIVE) == [5, 5, 5]
    assert _files_by_binding(allocation, RHS_ARCHIVE) == [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
    ]
    pairs = _pairs(allocation, RHS_ARCHIVE)
    assert len(set(pairs)) == 15


def test_rhs_archive_request_above_m_times_k_is_capped_with_one_warning(
    write_rhs_files: Callable[[int], str],
    matrix_bindings: list[SystemBinding],
    warning_messages: list[str],
) -> None:
    """A count above M*K is capped at M*K and reported once, naming rhs_archive."""
    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=_spec(RHS_ARCHIVE, write_rhs_files(5), 20),
        num_matrix_samples=3,
    )

    assert _counts(allocation, RHS_ARCHIVE) == [5, 5, 5]
    assert len(set(_pairs(allocation, RHS_ARCHIVE))) == 15
    assert len(warning_messages) == 1
    assert RHS_ARCHIVE in warning_messages[0]


def test_rhs_archive_single_matrix_several_bindings_rejected_before_glob_read(
    monkeypatch: pytest.MonkeyPatch,
    one_matrix_two_bindings: list[SystemBinding],
) -> None:
    """One matrix with several rhs_archive bindings is rejected before any glob read."""

    def _glob_must_not_be_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("archive glob read before the repeat check")

    monkeypatch.setattr(orchestration, "select_archive_files", _glob_must_not_be_read)
    with pytest.raises(ValueError, match="files would repeat across bindings"):
        _resolve_binding_strategy_counts(
            bindings=one_matrix_two_bindings,
            spec=_spec(RHS_ARCHIVE, "/fake/rhs_*.txt", 3),
            num_matrix_samples=1,
        )


def test_solution_archive_with_solution_source_and_glob_is_rejected(
    write_solution_files: Callable[[int], str],
    matrix_bindings: list[SystemBinding],
) -> None:
    """A per-binding solution_path and a solutions_glob are both sources: reject the mix.

    Passing the glob through unchanged would let the archive draw files outside the
    (matrix, file) assignment, so the combination is an explicit error.
    """
    with pytest.raises(ValueError, match="solution_archive"):
        _resolve_binding_strategy_counts(
            bindings=matrix_bindings,
            spec=_spec(SOLUTION_ARCHIVE, write_solution_files(5), 3),
            num_matrix_samples=3,
            has_solution_source=True,
        )


def test_rhs_archive_with_solution_source_and_glob_is_rejected(
    write_rhs_files: Callable[[int], str],
    matrix_bindings: list[SystemBinding],
) -> None:
    """rhs_archive with a per-binding solution source and rhs_glob is rejected the same way."""
    with pytest.raises(ValueError, match="rhs_archive"):
        _resolve_binding_strategy_counts(
            bindings=matrix_bindings,
            spec=_spec(RHS_ARCHIVE, write_rhs_files(5), 3),
            num_matrix_samples=3,
            has_solution_source=True,
        )


@pytest.mark.parametrize("strategy", ARCHIVE_STRATEGIES)
def test_archive_pairs_never_repeat_across_matrices(
    strategy: str,
    write_solution_files: Callable[[int], str],
    write_rhs_files: Callable[[int], str],
    matrix_bindings: list[SystemBinding],
) -> None:
    """Every (matrix, file) pair is distinct for each archive strategy, for several counts."""
    glob = write_rhs_files(4) if strategy == RHS_ARCHIVE else write_solution_files(4)
    for count in (1, 4, 7, 12, -1):
        allocation = _resolve_binding_strategy_counts(
            bindings=matrix_bindings,
            spec=_spec(strategy, glob, count),
            num_matrix_samples=3,
        )
        pairs = _pairs(allocation, strategy)
        assert len(set(pairs)) == len(pairs), f"{strategy} count={count} repeated a pair"


def test_both_archives_in_one_build_never_repeat_pairs(
    write_solution_files: Callable[[int], str],
    write_rhs_files: Callable[[int], str],
    matrix_bindings: list[SystemBinding],
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
                SOLUTION_ARCHIVE: {"solutions_glob": write_solution_files(4)},
                RHS_ARCHIVE: {"rhs_glob": write_rhs_files(4)},
            },
        ),
        replacement=False,
    )

    allocation = _resolve_binding_strategy_counts(
        bindings=matrix_bindings,
        spec=spec,
        num_matrix_samples=3,
    )

    for strategy in ARCHIVE_STRATEGIES:
        pairs = _pairs(allocation, strategy)
        assert len(pairs) == sum(_counts(allocation, strategy))
        assert len(set(pairs)) == len(pairs), f"{strategy} repeated a (matrix, file) pair"
