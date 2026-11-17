"""The offline break-even count N* of each method pair, from hand-written observations."""

from __future__ import annotations

import math

import pytest

from neuralls.domain.benchmarking.breakeven_full import (
    AlwaysCheaper,
    BreakEvenFull,
    BreaksEven,
    CheaperUntil,
    FullBreakEvenState,
    Never,
    breakeven_count,
    breakeven_full,
    count_bounds,
)
from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    Known,
    NotApplicable,
    OfflineStages,
    Unavailable,
)
from neuralls.domain.solver.models.statistics import Interval
from neuralls.shared.types import CostProvenance, PreconditionerName
from tests.observation_support import solved_row, synthetic_observations

CLASSICAL = PreconditionerName("classical")
FITTED = PreconditionerName("fitted")
SECOND = PreconditionerName("second")
UNRECORDED = PreconditionerName("unrecorded")
N_MATRICES = 5
OFFLINE_S = {CLASSICAL: 0.0, FITTED: 10.0, SECOND: 4.0}


def _fit(seconds: float) -> OfflineStages:
    return OfflineStages(NotApplicable(), Known(seconds, None, CostProvenance.HISTORICAL))


@pytest.fixture
def offline() -> dict[PreconditionerName, OfflineStages]:
    """A classical method (C = 0), two fitted ones (C = 10, 4) and one whose fit went unrecorded."""
    return {
        CLASSICAL: NO_OFFLINE_COST,
        FITTED: _fit(OFFLINE_S[FITTED]),
        SECOND: _fit(OFFLINE_S[SECOND]),
        UNRECORDED: OfflineStages(NotApplicable(), Unavailable("no fit run")),
    }


@pytest.mark.parametrize(
    ("offline_surplus_s", "saving_s", "expected"),
    [
        pytest.param(10.0, 2.0, BreaksEven(5.0), id="dearer-offline-faster-online"),
        pytest.param(-4.0, 2.0, AlwaysCheaper(), id="cheaper-offline-faster-online"),
        pytest.param(0.0, 2.0, AlwaysCheaper(), id="same-offline-faster-online"),
        pytest.param(-4.0, 0.0, AlwaysCheaper(), id="cheaper-offline-same-online"),
        pytest.param(-6.0, -3.0, CheaperUntil(2.0), id="cheaper-offline-slower-online"),
        pytest.param(10.0, -2.0, Never(), id="dearer-offline-slower-online"),
        pytest.param(0.0, 0.0, Never(), id="same-offline-same-online"),
    ],
)
def test_breakeven_count_state_follows_the_signs_of_offline_surplus_and_saving(
    offline_surplus_s: float, saving_s: float, expected: BreakEvenFull
) -> None:
    assert breakeven_count(offline_surplus_s, saving_s) == expected


@pytest.mark.parametrize(
    ("offline_surplus_s", "saving", "expected"),
    [
        pytest.param(10.0, (1.0, 2.0), (5.0, 10.0), id="cheaper-from-ends-swap"),
        pytest.param(10.0, (-1.0, 2.0), (5.0, math.inf), id="cheaper-from-reaching-never"),
        pytest.param(-10.0, (-2.0, -1.0), (5.0, 10.0), id="cheaper-until-ends-kept"),
        pytest.param(-10.0, (-2.0, 1.0), (5.0, math.inf), id="cheaper-until-reaching-always"),
    ],
)
def test_count_bounds_map_the_saving_interval_through_the_monotone_count(
    offline_surplus_s: float, saving: tuple[float, float], expected: tuple[float, float]
) -> None:
    assert count_bounds(offline_surplus_s, Interval(*saving, 0.95)) == expected


def test_breakeven_full_charges_both_offline_costs_and_skips_unknown_ones(
    offline: dict[PreconditionerName, OfflineStages],
) -> None:
    seconds = {CLASSICAL: 3.0, FITTED: 1.0, SECOND: 4.0, UNRECORDED: 0.5}
    observations = synthetic_observations(
        [
            solved_row(method, f"s{matrix}", matrix, online_s)
            for matrix in range(N_MATRICES)
            for method, online_s in seconds.items()
        ]
    )

    table = breakeven_full(observations, offline)

    def row(
        method: PreconditionerName,
        other: PreconditionerName,
        state: FullBreakEvenState,
        count: float | None,
        bounds: tuple[float, float],
    ) -> dict[str, object]:
        return {
            "method": method,
            "reference_method": other,
            "offline_s": OFFLINE_S[method],
            "reference_offline_s": OFFLINE_S[other],
            "online_mean_s": seconds[method],
            "reference_online_mean_s": seconds[other],
            "break_even": state.value,
            "breakeven_full": count,
            "ci_low": bounds[0],
            "ci_high": bounds[1],
            "n_pairs": N_MATRICES,
            "n_matrices": N_MATRICES,
        }

    assert table.to_dicts() == [
        pytest.approx(expected)
        for expected in (
            row(FITTED, CLASSICAL, FullBreakEvenState.BREAKS_EVEN, 5.0, (5.0, 5.0)),
            row(FITTED, SECOND, FullBreakEvenState.BREAKS_EVEN, 2.0, (2.0, 2.0)),
            row(SECOND, CLASSICAL, FullBreakEvenState.NEVER, None, (math.inf, math.inf)),
            row(SECOND, FITTED, FullBreakEvenState.CHEAPER_UNTIL, 2.0, (2.0, 2.0)),
        )
    ]


def test_breakeven_full_without_a_common_pair_has_no_means(
    offline: dict[PreconditionerName, OfflineStages],
) -> None:
    observations = synthetic_observations(
        [solved_row(FITTED, "s0", 0, 1.0), solved_row(CLASSICAL, "s1", 1, 3.0)]
    )

    table = breakeven_full(observations, offline)

    assert table.select("method", "reference_method", "break_even", "n_pairs").rows() == [
        (FITTED, CLASSICAL, FullBreakEvenState.NO_COMMON_SYSTEMS.value, 0)
    ]
