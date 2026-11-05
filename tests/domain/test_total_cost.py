"""`total_cost_seconds` adds offline to online, and states when offline is not known."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    Known,
    NotApplicable,
    OfflineStages,
    Unavailable,
)
from neuralls.domain.solver.reporting.total_cost import (
    Complete,
    OfflineUnavailable,
    TotalCost,
    total_cost_seconds,
)
from neuralls.shared.types import CostProvenance

ONLINE_S = 7.5


def _known(seconds: float) -> Known:
    return Known(seconds, None, CostProvenance.HISTORICAL)


def test_total_is_offline_plus_online() -> None:
    offline = OfflineStages(generation=_known(10.0), training=_known(90.0))

    assert total_cost_seconds(offline, ONLINE_S) == Complete(107.5)


def test_a_method_with_no_offline_stage_costs_its_online_time() -> None:
    assert total_cost_seconds(NO_OFFLINE_COST, ONLINE_S) == Complete(ONLINE_S)


def test_a_stage_that_does_not_apply_is_zero_not_unavailable() -> None:
    offline = OfflineStages(generation=_known(10.0), training=NotApplicable())

    assert total_cost_seconds(offline, ONLINE_S) == Complete(17.5)


@pytest.mark.parametrize(
    "offline",
    [
        OfflineStages(_known(10.0), Unavailable("no run id")),
        OfflineStages(Unavailable("no manifest"), _known(90.0)),
        OfflineStages(Unavailable("no manifest"), Unavailable("no run id")),
    ],
)
def test_an_unavailable_stage_is_never_read_as_zero(offline: OfflineStages) -> None:
    total: TotalCost = total_cost_seconds(offline, ONLINE_S)

    assert total == OfflineUnavailable(online_s=ONLINE_S)
