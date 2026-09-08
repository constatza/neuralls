"""Tests for multi-matrix orchestration allocation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.orchestration import (
    _allocate_strategy_counts_across_bindings,
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


class _FakeRng:
    """Deterministic stand-in for allocation tests."""

    def __init__(self, draws: list[int]) -> None:
        self._draws = np.asarray(draws, dtype=np.int64)

    def integers(self, low: int, high: int | None = None, size: int | None = None) -> np.ndarray:
        assert low == 0
        assert high is not None
        assert size == self._draws.size
        return self._draws


@pytest.fixture
def three_bindings() -> list[SystemBinding]:
    """Three independent matrix bindings."""
    return [
        SystemBinding(sample_id=0, matrix_sample_id=0),
        SystemBinding(sample_id=1, matrix_sample_id=1),
        SystemBinding(sample_id=2, matrix_sample_id=2),
    ]


def test_allocate_strategy_counts_without_replacement_is_even(
    three_bindings: list[SystemBinding],
) -> None:
    allocations = _allocate_strategy_counts_across_bindings(
        count=5,
        bindings=three_bindings,
        replacement=False,
        rng=np.random.default_rng(0),
    )

    assert allocations == [2, 2, 1]


def test_allocate_strategy_counts_with_replacement_uses_sampled_bindings(
    three_bindings: list[SystemBinding],
) -> None:
    allocations = _allocate_strategy_counts_across_bindings(
        count=5,
        bindings=three_bindings,
        replacement=True,
        rng=_FakeRng([2, 2, 0, 2, 1]),
    )

    assert allocations == [1, 1, 3]


def test_resolve_binding_strategy_counts_rejects_unsupported_replacement(
    three_bindings: list[SystemBinding],
) -> None:
    with pytest.raises(ValueError, match="does not support matrix replacement"):
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
            has_rhs_source=False,
            num_matrix_samples=3,
        )


def test_resolve_binding_strategy_counts_rejects_finite_trace_replacement(
    three_bindings: list[SystemBinding],
) -> None:
    with pytest.raises(ValueError, match="finite external source"):
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
            has_rhs_source=True,
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
    # cg_iters=3 → rows_per_system=4; samples=8 → 2 base systems, 8 trace pairs.
    # Bug: shuffle indexed row_kind_codes (len=8) with 2 base-system indices → truncated to len=2.
    result = _generate_mixture_with_metadata(
        spd_matrix,
        MixtureSpec(
            counts={"gaussian_residuals": 8},
            seed=0,
            shuffle=True,
            strategy_overrides={"gaussian_residuals": {"cg_iters": 3}},
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
    # gaussian_residuals:4 with cg_iters=1 → 2 base systems × 2 rows = 4 trace rows
    #   each system: [iter0=STANDARD, iter1=CG_INTERNAL]
    result = _generate_mixture_with_metadata(
        spd_matrix,
        MixtureSpec(
            counts={"gaussian_forward": 3, "gaussian_residuals": 4},
            seed=0,
            shuffle=False,
            strategy_overrides={"gaussian_residuals": {"cg_iters": 1}},
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
                "gaussian_residuals": {"cg_iters": 2, "seed": 42},
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
                    "cg_iters": 2,
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
            has_rhs_source=False,
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
    with pytest.raises(ValueError, match="does not support matrix replacement"):
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
            has_rhs_source=False,
            num_matrix_samples=3,
        )


def test_resolve_binding_strategy_counts_divides_all_samples_across_bindings(
    tmp_path: Path,
    three_bindings: list[SystemBinding],
) -> None:
    """samples=-1 ("all") must resolve to the real archive size and divide across bindings.

    Regression test: before this fix, -1 was replicated unchanged to every binding
    (`_allocate_strategy_counts_across_bindings`'s `count == _ALL_SAMPLES` branch), so
    each binding loaded the *entire* archive independently instead of the archive being
    split across bindings the way an explicit positive count already is — a
    cartesian-product blowup (num_bindings x archive_size RHS computations).
    """
    for idx in range(5):
        np.savetxt(tmp_path / f"solution_{idx:03d}.txt", np.full(4, float(idx)))
    glob_pattern = str(tmp_path / "solution_*.txt")

    allocation = _resolve_binding_strategy_counts(
        bindings=three_bindings,
        spec=DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": -1},
                seed=0,
                strategy_overrides={"solution_archive": {"solutions_glob": glob_pattern}},
            ),
            replacement=False,
        ),
        has_rhs_source=False,
        num_matrix_samples=3,
    )

    allocated = [counts.get("solution_archive", 0) for counts in allocation.counts]
    assert allocated == [2, 2, 1]
    assert sum(allocated) == 5  # the real archive size, not -1 replicated per binding

    # Cumulative, disjoint offsets into one shared shuffle — binding 0 takes files
    # [0:2], binding 1 takes [2:4], binding 2 takes [4:5]; no binding reuses another's slice.
    skips = [skip.get("solution_archive", 0) for skip in allocation.skips]
    assert skips == [0, 2, 4]


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
            has_rhs_source=False,
            num_matrix_samples=3,
        )
