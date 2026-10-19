"""Tests for ``break_even``: the crossing of two ``T(n) = a + b * n`` cost lines."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.breakeven import break_even
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.models.statistics import (
    AlwaysCheaper,
    BreakEven,
    BreakEvenAt,
    CheaperOnlyBelow,
    NeverCheaper,
)


def _line(*, one_time: float, per_solved: float) -> ProjectedComparisonResult:
    """A fully converged projection with the given one-time cost ``a`` and recurring cost ``b``."""
    return ProjectedComparisonResult(
        preconditioner_name="m",
        n_systems=1,
        n_converged=1,
        setup_sharing=SetupSharingPolicy(share_setup=True, share_generation=True),
        setup_cost_total_seconds=one_time,
        generation_cost_total_seconds=None,
        total_solve_time_seconds=per_solved,
        total_iterations=1,
    )


@pytest.mark.parametrize(
    ("method", "reference", "expected"),
    [
        # a_m=10,b_m=1 vs a_r=0,b_r=3: 10 + n = 3n at n = 5.
        ((10.0, 1.0), (0.0, 3.0), BreakEvenAt(5.0)),
        # Lower recurring and not-higher one-time cost: cheaper from the start.
        ((0.0, 1.0), (5.0, 3.0), AlwaysCheaper()),
        ((5.0, 1.0), (5.0, 3.0), AlwaysCheaper()),
        # a_m=0,b_m=3 vs a_r=10,b_r=1: 3n = 10 + n at n = 5, cheaper only before it.
        ((0.0, 3.0), (10.0, 1.0), CheaperOnlyBelow(5.0)),
        # Higher recurring and higher one-time cost: never cheaper.
        ((10.0, 3.0), (0.0, 1.0), NeverCheaper()),
        # Equal recurring cost: decided by the one-time cost alone.
        ((0.0, 2.0), (4.0, 2.0), AlwaysCheaper()),
        ((4.0, 2.0), (4.0, 2.0), NeverCheaper()),
        ((5.0, 2.0), (4.0, 2.0), NeverCheaper()),
    ],
)
def test_break_even_variants(
    method: tuple[float, float], reference: tuple[float, float], expected: BreakEven
) -> None:
    result = break_even(
        _line(one_time=method[0], per_solved=method[1]),
        _line(one_time=reference[0], per_solved=reference[1]),
    )

    assert result == expected


def test_break_even_uses_per_solved_cost_of_an_unreliable_method() -> None:
    """A repeated-setup method that fails often pays its attempts per solved system."""
    unreliable = ProjectedComparisonResult(
        preconditioner_name="u",
        n_systems=5,
        n_converged=3,
        setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=False),
        setup_cost_total_seconds=5.0,
        generation_cost_total_seconds=None,
        total_solve_time_seconds=6.2,
        total_iterations=230,
    )
    reference = _line(one_time=0.0, per_solved=2.0)

    # b_u = (1.0 + 1.24) / 0.6 = 3.7333 > 2.0 and a_u = 0 = a_r: never cheaper.
    assert break_even(unreliable, reference) == NeverCheaper()
