"""`StageCost` variants: only a recorded cost has seconds; not applicable and unavailable do not."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.cost import (
    Known,
    NotApplicable,
    StageCost,
    Unavailable,
    recorded_seconds,
)
from neuralls.shared.types import CostProvenance


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (Known(5.0, None, CostProvenance.HISTORICAL), 5.0),
        (Known(0.0, None, CostProvenance.MEASURED), 0.0),
        (NotApplicable(), None),
        (Unavailable("no record"), None),
    ],
)
def test_recorded_seconds_exists_only_for_a_known_cost(
    cost: StageCost, expected: float | None
) -> None:
    assert recorded_seconds(cost) == expected


def test_unavailable_cost_has_no_number_to_misread() -> None:
    assert not hasattr(Unavailable("no record"), "wall_time_seconds")
