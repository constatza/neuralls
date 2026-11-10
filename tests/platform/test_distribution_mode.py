"""A distribution row's mode follows how many systems the method has."""

from __future__ import annotations

import pytest

from neuralls.platform.reporting.distribution_mode import (
    BoxAndSwarm,
    PointsOnly,
    RowLimits,
    RowMode,
    Violin,
    row_mode,
)

LIMITS = RowLimits(min_box_samples=5, min_violin_samples=30)


@pytest.mark.parametrize(
    ("n_samples", "expected"),
    [
        (0, PointsOnly()),
        (4, PointsOnly()),
        (5, BoxAndSwarm()),
        (29, BoxAndSwarm()),
        (30, Violin()),
        (1000, Violin()),
    ],
)
def test_the_mode_changes_exactly_at_the_limits(n_samples: int, expected: RowMode) -> None:
    assert row_mode(n_samples, LIMITS) == expected


def test_equal_limits_skip_the_box_and_swarm_mode() -> None:
    limits = RowLimits(min_box_samples=3, min_violin_samples=3)

    assert [row_mode(n, limits) for n in (2, 3)] == [PointsOnly(), Violin()]
