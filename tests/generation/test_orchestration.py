"""Tests for multi-matrix orchestration allocation helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation import orchestration
from neuralls.domain.generation.orchestration import (
    BindingAllocation,
    _generate_mixture_with_metadata,
    _resolve_binding_strategy_counts,
)
from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """Small SPD matrix for orchestration tests."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    return A.T @ A + np.eye(8)


@pytest.fixture
def three_bindings() -> list[SystemBinding]:
    """Three independent matrix bindings."""
    return [
        SystemBinding(sample_id=0, matrix_sample_id=0),
        SystemBinding(sample_id=1, matrix_sample_id=1),
        SystemBinding(sample_id=2, matrix_sample_id=2),
    ]


def test_resolve_binding_strategy_counts_rejects_unsupported_replacement(
    three_bindings: list[SystemBinding],
) -> None:
    with pytest.raises(ValueError, match="replacement = true is not supported"):
        _resolve_binding_strategy_counts(
            bindings=three_bindings,
            spec=DatasetSpec(
                mixture=MixtureSpec(
                    counts={"neutral_ones": 5},
                    seed=0,
                    strategy_overrides=None,
                ),
                replacement=True,
            ),
            num_matrix_samples=3,
        )


def test_resolve_binding_strategy_counts_rejects_finite_trace_replacement(
    three_bindings: list[SystemBinding],
) -> None:
    with pytest.raises(ValueError, match="replacement = true is not supported"):
        _resolve_binding_strategy_counts(
            bindings=three_bindings,
            spec=DatasetSpec(
                mixture=MixtureSpec(
                    counts={"residuals": 5},
                    seed=0,
                    strategy_overrides=None,
                ),
                replacement=True,
            ),
            num_matrix_samples=3,
        )


def test_generate_mixture_row_kind_codes_length_matches_trace_rows_after_shuffle(
    spd_matrix: np.ndarray,
    solver_overrides: dict,
) -> None:
    """row_kind_codes must align with error_traces.residuals, not with base-system count.

    gaussian_residuals produces N base systems but N*rows_per_system trace pairs.
    With shuffle=True the shuffle used to index row_kind_codes (trace-level, final_rows entries)
    with base-system-level indices (referenced_samples entries), silently truncating it.
    _finalize_payload then caught the mismatch: row_kind_codes.shape[0] != rhs_all.shape[0].
    """
    # stop=3, start=0 → rows_per_system=4; samples=8 → 2 base systems, 8 trace pairs.
    # Bug: shuffle indexed row_kind_codes (len=8) with 2 base-system indices → truncated to len=2.
    result = _generate_mixture_with_metadata(
        spd_matrix,
        MixtureSpec(
            counts={"gaussian_residuals": 8},
            seed=0,
            shuffle=True,
            strategy_overrides={"gaussian_residuals": {"stop": 3, "start": 0}},
            solver_overrides=solver_overrides,
        ),
    )

    assert result.row_kind_codes.shape[0] == result.rhs.shape[0]


def test_mixed_strategy_row_kind_codes_concatenated_correctly(
    spd_matrix: np.ndarray,
    solver_overrides: dict,
) -> None:
    """row_kind_codes must cover every row in order: forward rows first, then trace rows."""
    from neuralls.shared.enum_codecs import decode_row_kind_array
    from neuralls.shared.types import RowKind

    # gaussian_forward:3 → 3 STANDARD rows
    # gaussian_residuals:4 with stop=1, start=0 → 2 base systems × 2 rows = 4 trace rows
    #   each system: [iter0=STANDARD, iter1=CG_INTERNAL]
    result = _generate_mixture_with_metadata(
        spd_matrix,
        MixtureSpec(
            counts={"gaussian_forward": 3, "gaussian_residuals": 4},
            seed=0,
            shuffle=False,
            strategy_overrides={"gaussian_residuals": {"stop": 1, "start": 0}},
            solver_overrides=solver_overrides,
        ),
    )

    assert result.rhs.shape[0] == 7
    assert result.row_kind_codes.shape[0] == 7

    kinds = decode_row_kind_array(result.row_kind_codes)
    assert kinds[:3] == (RowKind.STANDARD,) * 3
    assert kinds[3] == RowKind.STANDARD  # iter 0, system 0
    assert kinds[4] == RowKind.CG_INTERNAL  # iter 1, system 0
    assert kinds[5] == RowKind.STANDARD  # iter 0, system 1
    assert kinds[6] == RowKind.CG_INTERNAL  # iter 1, system 1


def test_gaussian_split_mix_preserves_requested_total_rows(
    spd_matrix: np.ndarray, solver_overrides: dict
) -> None:
    """A residual/pure Gaussian split uses exact row budgets for both strategies."""
    from neuralls.shared.enum_codecs import decode_row_kind_array
    from neuralls.shared.types import RowKind

    result = _generate_mixture_with_metadata(
        spd_matrix,
        MixtureSpec(
            counts={"gaussian_residuals": 5, "gaussian_forward": 5},
            seed=0,
            shuffle=False,
            strategy_overrides={
                "gaussian_residuals": {"stop": 2, "start": 0, "seed": 42},
                "gaussian_forward": {"seed": 43},
            },
            solver_overrides=solver_overrides,
        ),
    )

    assert result.rhs.shape[0] == 10
    assert result.solutions.shape == result.rhs.shape
    assert result.row_kind_codes.shape[0] == 10
    assert decode_row_kind_array(result.row_kind_codes) == (
        RowKind.STANDARD,
        RowKind.CG_INTERNAL,
        RowKind.CG_INTERNAL,
        RowKind.STANDARD,
        RowKind.CG_INTERNAL,
        RowKind.STANDARD,
        RowKind.STANDARD,
        RowKind.STANDARD,
        RowKind.STANDARD,
        RowKind.STANDARD,
    )


def test_archive_split_mix_uses_solution_archive_skip(
    tmp_path: Path,
    spd_matrix: np.ndarray,
    solver_overrides: dict,
) -> None:
    """Archive-backed split rows can skip residual base systems to avoid duplicate pairs."""
    vectors = []
    for idx in range(4):
        vector = np.full(spd_matrix.shape[0], float(idx + 1), dtype=np.float64)
        path = tmp_path / f"solution_{idx:03d}.txt"
        np.savetxt(path, vector)
        vectors.append(vector)

    glob_pattern = str(tmp_path / "solution_*.txt")
    result = _generate_mixture_with_metadata(
        spd_matrix,
        MixtureSpec(
            counts={"residuals": 3, "solution_archive": 2},
            seed=0,
            shuffle=False,
            strategy_overrides={
                "residuals": {
                    "stop": 2,
                    "start": 0,
                    "solutions_glob": glob_pattern,
                    "shuffle": False,
                },
                "solution_archive": {
                    "solutions_glob": glob_pattern,
                    "shuffle": False,
                    "skip": 1,
                },
            },
            solver_overrides=solver_overrides,
        ),
    )

    assert result.rhs.shape[0] == 5
    np.testing.assert_array_equal(result.solutions[3:], np.vstack(vectors[1:3]))


def test_resolve_binding_strategy_counts_rejects_single_multi_matrix_mix(
    three_bindings: list[SystemBinding],
) -> None:
    """Mixing single-matrix and multi-matrix strategies with multiple matrices must fail."""
    with pytest.raises(ValueError, match="Cannot mix single-matrix strategies"):
        _resolve_binding_strategy_counts(
            bindings=three_bindings,
            spec=DatasetSpec(
                mixture=MixtureSpec(
                    counts={"gaussian_forward": 5, "solution_archive": 5},
                    seed=0,
                    strategy_overrides={"solution_archive": {"solutions_glob": "/fake/*.txt"}},
                ),
                replacement=False,
            ),
            num_matrix_samples=3,
        )


def test_resolve_binding_strategy_counts_rejects_all_samples_with_replacement(
    three_bindings: list[SystemBinding],
) -> None:
    """samples=-1 must not be invisible to the replacement guard.

    Regression test: `_ALL_SAMPLES` (-1) used to be excluded from the `active`
    strategy list (`count > 0` filter), so `solution_archive: -1` combined with
    `replacement=True` silently skipped the "does not support matrix
    replacement allocation" guard instead of raising.
    """
    with pytest.raises(ValueError, match="replacement = true is not supported"):
        _resolve_binding_strategy_counts(
            bindings=three_bindings,
            spec=DatasetSpec(
                mixture=MixtureSpec(
                    counts={"solution_archive": -1},
                    seed=0,
                    strategy_overrides={"solution_archive": {"solutions_glob": "/fake/*.txt"}},
                ),
                replacement=True,
            ),
            num_matrix_samples=3,
        )


def _archive_spec(glob_pattern: str, count: int) -> DatasetSpec:
    """Dataset spec with one solution_archive count over the given glob."""
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={"solution_archive": count},
            seed=0,
            strategy_overrides={"solution_archive": {"solutions_glob": glob_pattern}},
        ),
        replacement=False,
    )


def _files_by_binding(allocation: BindingAllocation, strategy_name: str) -> list[list[int]]:
    """Explicit archive file indices per binding (empty list when a binding has none)."""
    return [list(indices.get(strategy_name, ())) for indices in allocation.file_indices]


def _archive_pairs(allocation: BindingAllocation) -> list[tuple[int, int]]:
    """(binding, file) pairs; one binding per matrix in these multi-matrix fixtures."""
    return [
        (binding_idx, file_idx)
        for binding_idx, files in enumerate(_files_by_binding(allocation, "solution_archive"))
        for file_idx in files
    ]


def test_resolve_archive_counts_and_files_are_distinct_pairs(
    write_solution_files: Callable[[int], str],
    three_bindings: list[SystemBinding],
) -> None:
    """Archive units are spread over the pool with cyclic, non-repeating (matrix, file) pairs."""
    glob_pattern = write_solution_files(5)

    allocation = _resolve_binding_strategy_counts(
        bindings=three_bindings,
        spec=_archive_spec(glob_pattern, 5),
        num_matrix_samples=3,
    )

    assert [counts["solution_archive"] for counts in allocation.counts] == [2, 2, 1]
    assert _files_by_binding(allocation, "solution_archive") == [[0, 1], [1, 2], [2]]
    pairs = _archive_pairs(allocation)
    assert len(pairs) == 5
    assert len(set(pairs)) == 5


def test_archive_base_split_uses_full_pool_without_repeats(
    write_solution_files: Callable[[int], str],
    three_bindings: list[SystemBinding],
) -> None:
    """A count equal to the full M*K pool gives matrix i ceil((50 - i) / 3) distinct files."""
    glob_pattern = write_solution_files(50)

    allocation = _resolve_binding_strategy_counts(
        bindings=three_bindings,
        spec=_archive_spec(glob_pattern, 50),
        num_matrix_samples=3,
    )

    assert [counts["solution_archive"] for counts in allocation.counts] == [17, 17, 16]
    pairs = _archive_pairs(allocation)
    assert len(set(pairs)) == 50


def test_archive_all_samples_emits_every_pair_once(
    write_solution_files: Callable[[int], str],
    three_bindings: list[SystemBinding],
) -> None:
    """samples = -1 on an archive emits all M*K (matrix, file) pairs, each exactly once."""
    glob_pattern = write_solution_files(5)

    allocation = _resolve_binding_strategy_counts(
        bindings=three_bindings,
        spec=_archive_spec(glob_pattern, -1),
        num_matrix_samples=3,
    )

    assert [counts["solution_archive"] for counts in allocation.counts] == [5, 5, 5]
    assert _files_by_binding(allocation, "solution_archive") == [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
    ]
    pairs = _archive_pairs(allocation)
    assert len(pairs) == 15
    assert len(set(pairs)) == 15
    assert [binding for binding, _ in pairs] == [0] * 5 + [1] * 5 + [2] * 5


def test_archive_request_above_m_times_k_is_capped_with_one_warning(
    write_solution_files: Callable[[int], str],
    three_bindings: list[SystemBinding],
    warning_messages: list[str],
) -> None:
    """A count above M*K is capped at M*K and reported once, naming both counts."""
    glob_pattern = write_solution_files(5)

    allocation = _resolve_binding_strategy_counts(
        bindings=three_bindings,
        spec=_archive_spec(glob_pattern, 20),
        num_matrix_samples=3,
    )

    assert [counts["solution_archive"] for counts in allocation.counts] == [5, 5, 5]
    assert len(set(_archive_pairs(allocation))) == 15
    assert len(warning_messages) == 1
    assert "solution_archive" in warning_messages[0]
    assert "20" in warning_messages[0]
    assert "15" in warning_messages[0]


@pytest.mark.parametrize("requested", [15, 14])
def test_archive_request_at_or_below_m_times_k_has_no_warning(
    write_solution_files: Callable[[int], str],
    three_bindings: list[SystemBinding],
    warning_messages: list[str],
    requested: int,
) -> None:
    """A count at or below M*K is emitted in full with no warning."""
    glob_pattern = write_solution_files(5)

    allocation = _resolve_binding_strategy_counts(
        bindings=three_bindings,
        spec=_archive_spec(glob_pattern, requested),
        num_matrix_samples=3,
    )

    total = sum(counts["solution_archive"] for counts in allocation.counts)
    assert total == requested
    assert warning_messages == []


def test_archive_single_matrix_several_bindings_is_rejected_before_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One matrix with several archive bindings would repeat files: reject before any glob read."""

    def _glob_must_not_be_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("archive glob read before the repeat check")

    monkeypatch.setattr(orchestration, "select_archive_files", _glob_must_not_be_read)
    bindings = [
        SystemBinding(sample_id=0, matrix_sample_id=0, rhs_sample_id=0),
        SystemBinding(sample_id=1, matrix_sample_id=0, rhs_sample_id=1),
    ]

    with pytest.raises(ValueError, match="files would repeat across bindings"):
        _resolve_binding_strategy_counts(
            bindings=bindings,
            spec=_archive_spec("/fake/solution_*.txt", 3),
            num_matrix_samples=1,
        )


def test_archive_single_matrix_single_binding_still_works(
    write_solution_files: Callable[[int], str],
) -> None:
    """A single matrix with one binding keeps the requested archive count."""
    glob_pattern = write_solution_files(5)

    allocation = _resolve_binding_strategy_counts(
        bindings=[SystemBinding(sample_id=0, matrix_sample_id=0)],
        spec=_archive_spec(glob_pattern, 3),
        num_matrix_samples=1,
    )

    assert [counts["solution_archive"] for counts in allocation.counts] == [3]


def test_resolve_generated_remainder_is_fully_allocated(
    three_bindings: list[SystemBinding],
    warning_messages: list[str],
) -> None:
    """A generated count with a remainder is split without dropping any samples."""
    allocation = _resolve_binding_strategy_counts(
        bindings=three_bindings,
        spec=DatasetSpec(
            mixture=MixtureSpec(counts={"gaussian_forward": 7}, seed=0),
            replacement=False,
        ),
        num_matrix_samples=3,
    )

    counts = [counts["gaussian_forward"] for counts in allocation.counts]
    assert sorted(counts) == [2, 2, 3]
    assert sum(counts) == 7
    assert warning_messages == []


def test_resolve_binding_strategy_counts_rejects_unresolvable_all_samples(
    three_bindings: list[SystemBinding],
) -> None:
    """samples=-1 across multiple bindings must fail fast without a resolvable glob."""
    with pytest.raises(ValueError, match="has no 'solutions_glob'"):
        _resolve_binding_strategy_counts(
            bindings=three_bindings,
            spec=DatasetSpec(
                mixture=MixtureSpec(
                    counts={"gaussian_forward": -1},
                    seed=0,
                    strategy_overrides=None,
                ),
                replacement=False,
            ),
            num_matrix_samples=3,
        )
