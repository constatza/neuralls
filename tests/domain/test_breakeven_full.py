"""The offline break-even count N* of each method pair, from hand-written observations."""

from __future__ import annotations

import math

import pytest

from neuralls.domain.benchmarking.breakeven_full import breakeven_full
from neuralls.domain.benchmarking.cost_curves import BreakEvenState
from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    Known,
    NotApplicable,
    OfflineStages,
    StageCost,
    Unavailable,
)
from neuralls.shared.types import CostProvenance, PreconditionerName
from tests.observation_support import solved_row, synthetic_observations

FITTED = PreconditionerName("fitted")
CLASSICAL = PreconditionerName("classical")
OFFLINE_S = 10.0


def _offline(fit: StageCost) -> dict[PreconditionerName, OfflineStages]:
    return {FITTED: OfflineStages(NotApplicable(), fit), CLASSICAL: NO_OFFLINE_COST}


def _row(
    n_matrices: int, fitted_s: float, classical_s: float, state: BreakEvenState
) -> dict[str, object]:
    return {
        "method": FITTED,
        "reference_method": CLASSICAL,
        "offline_s": OFFLINE_S,
        "online_mean_s": fitted_s,
        "reference_online_mean_s": classical_s,
        "break_even": state.value,
        "n_pairs": n_matrices,
        "n_matrices": n_matrices,
    }


@pytest.mark.parametrize(
    ("n_matrices", "fitted_s", "classical_s", "fit", "expected"),
    [
        pytest.param(
            5,
            1.0,
            3.0,
            Known(OFFLINE_S, None, CostProvenance.HISTORICAL),
            [
                _row(5, 1.0, 3.0, BreakEvenState.BREAKS_EVEN)
                | {"breakeven_full": 5.0, "ci_low": 5.0, "ci_high": 5.0}
            ],
            id="faster-per-pair-breaks-even-at-offline-over-saving",
        ),
        pytest.param(
            5,
            3.0,
            1.0,
            Known(OFFLINE_S, None, CostProvenance.HISTORICAL),
            [
                _row(5, 3.0, 1.0, BreakEvenState.NEVER)
                | {"breakeven_full": None, "ci_low": math.inf, "ci_high": math.inf}
            ],
            id="no-saving-never-breaks-even",
        ),
        pytest.param(
            5,
            1.0,
            3.0,
            Unavailable("no fit run"),
            [],
            id="unavailable-offline-cost-has-no-row",
        ),
        pytest.param(
            2,
            1.0,
            3.0,
            Known(OFFLINE_S, None, CostProvenance.HISTORICAL),
            [
                _row(2, 1.0, 3.0, BreakEvenState.BREAKS_EVEN)
                | {"breakeven_full": 5.0, "ci_low": None, "ci_high": None}
            ],
            id="fewer-than-five-matrices-have-no-interval",
        ),
    ],
)
def test_breakeven_full_is_offline_over_the_online_saving_per_pair(
    n_matrices: int,
    fitted_s: float,
    classical_s: float,
    fit: StageCost,
    expected: list[dict[str, object]],
) -> None:
    observations = synthetic_observations(
        [
            row
            for matrix in range(n_matrices)
            for row in (
                solved_row(FITTED, f"s{matrix}", matrix, fitted_s),
                solved_row(CLASSICAL, f"s{matrix}", matrix, classical_s),
            )
        ]
    )

    table = breakeven_full(observations, _offline(fit))

    assert table.to_dicts() == [pytest.approx(row) for row in expected]
