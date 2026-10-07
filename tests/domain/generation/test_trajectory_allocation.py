"""Trajectory rows are allocated as whole base systems, with overshoot trimmed once.

Every trajectory strategy here keeps K_rows = 4 rows per base system (window 1..4,
K = stop - start + 1 = 4; iterate 0 is the base pair and is never emitted).
Hand derivations are in the comments next to each expected multiset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from neuralls.domain.generation.batch_plan import BindingAllocation
from neuralls.domain.generation.binding_allocation import _resolve_binding_strategy_counts
from neuralls.domain.generation.helpers import required_trace_systems
from neuralls.domain.generation.runner import rows_per_base_system
from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec
from neuralls.domain.generation.step_window import StepWindow

from .conftest import TRACE_WINDOW_OVERRIDES

TRACE_WINDOW = StepWindow(start=1, stop=4, step=1)
TRACE_ROWS_PER_SYSTEM = 4
TRACE_STRATEGIES = (
    "residuals",
    "gaussian_residuals",
    "smoother_filtered_probes",
)


def _trace_mixture(strategy: str, rows: int, *, glob: str | None = None) -> MixtureSpec:
    overrides: dict[str, dict[str, object]] = {
        strategy: {**TRACE_WINDOW_OVERRIDES, **({"solutions_glob": glob} if glob else {})}
    }
    return MixtureSpec(counts={strategy: rows}, seed=0, strategy_overrides=overrides)


def _allocate(
    bindings: list[SystemBinding], mixture: MixtureSpec, num_matrices: int
) -> BindingAllocation:
    return _resolve_binding_strategy_counts(
        bindings=bindings,
        spec=DatasetSpec(mixture=mixture),
        num_matrix_samples=num_matrices,
    )


def _rows(allocation: BindingAllocation, strategy: str) -> list[int]:
    return [counts[strategy] for counts in allocation.counts if strategy in counts]


def test_rows_to_base_systems_uses_ceil_per_strategy_window() -> None:
    """B = ceil(R / K_rows) for every trace strategy; K_rows comes from its window."""
    for strategy in TRACE_STRATEGIES:
        assert rows_per_base_system(strategy, {**TRACE_WINDOW_OVERRIDES, "samples": 1000}) == 4
    assert rows_per_base_system("gaussian_forward", {}) == 1
    assert [required_trace_systems(rows, window=TRACE_WINDOW) for rows in (1000, 1001, 1002)] == [
        250,
        251,
        251,
    ]


def test_residuals_split_is_on_base_systems(
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    """60 matrices, 1000 rows: 250 base systems, 10 matrices get 5 (20 rows), 50 get 4."""
    allocation = _allocate(make_bindings(60, 1), _trace_mixture("residuals", 1000), 60)

    rows = _rows(allocation, "residuals")
    assert Counter(rows) == {20: 10, 16: 50}
    assert sum(rows) == 1000
    assert Counter(required_trace_systems(r, window=TRACE_WINDOW) for r in rows) == {5: 10, 4: 50}


def test_residuals_overshoot_is_trimmed_from_last_unit_owner(
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    """R = 1002: B = 251 = 60 * 4 + 11, so 2 rows overshoot and are trimmed from one owner.

    Which matrix owns the trimmed unit depends on the seed, so the test checks the
    invariants instead of a fixed multiset: the rows sum to the request, every matrix
    keeps the base systems it was allocated, and exactly one matrix is short by the
    overshoot.
    """
    allocation = _allocate(make_bindings(60, 1), _trace_mixture("residuals", 1002), 60)

    rows = _rows(allocation, "residuals")
    assert sum(rows) == 1002
    base_systems = [required_trace_systems(r, window=TRACE_WINDOW) for r in rows]
    assert sum(base_systems) == 251
    shortfalls = [TRACE_ROWS_PER_SYSTEM * b - r for b, r in zip(base_systems, rows, strict=True)]
    assert sorted(shortfalls) == [0] * 59 + [2]


def test_non_multiple_request_small_example(
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    """R = 10 over 3 matrices: B = 3, one unit each, overshoot O = 12 - 10 = 2."""
    allocation = _allocate(make_bindings(3, 1), _trace_mixture("residuals", 10), 3)

    rows = _rows(allocation, "residuals")
    assert Counter(rows) == {4: 2, 2: 1}
    assert sum(rows) == 10


def test_trajectory_archive_units_are_distinct_base_systems(
    make_bindings: Callable[[int, int], list[SystemBinding]],
    make_archive_glob: Callable[[int], str],
) -> None:
    """3 matrices, 5 files, R = 8: B = 2 base systems at (matrix, file) (0, 0) and (1, 1)."""
    glob = make_archive_glob(5)
    allocation = _allocate(make_bindings(3, 1), _trace_mixture("residuals", 8, glob=glob), 3)

    assert allocation.file_indices[0]["residuals"] == (0,)
    assert allocation.file_indices[1]["residuals"] == (1,)
    assert _rows(allocation, "residuals") == [4, 4]


def test_archive_cap_cuts_base_systems_not_expansion(
    make_bindings: Callable[[int, int], list[SystemBinding]],
    make_archive_glob: Callable[[int], str],
    warning_messages: list[str],
) -> None:
    """M = 3, K = 2 files: R = 40 asks for B = 10, only M*K = 6 base systems exist: 24 rows."""
    glob = make_archive_glob(2)
    allocation = _allocate(make_bindings(3, 1), _trace_mixture("residuals", 40, glob=glob), 3)

    assert sum(_rows(allocation, "residuals")) == 24
    assert required_trace_systems(24, window=TRACE_WINDOW) == 6
    assert len(warning_messages) == 1
    assert "40 rows" in warning_messages[0]
    assert "10 base systems" in warning_messages[0]
    assert "24 rows" in warning_messages[0]
