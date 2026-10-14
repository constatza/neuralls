"""Tests for the pure helpers in ``scripts/pod2g_amg_convergence_study.py``.

Only the argument-free logic (grid parsing, nearest-realized-c matching) is
covered here — the sweep functions themselves require a real system matrix
and POD-2G snapshot dataset and are exercised by running the script directly
per ``docs/plan.md``'s Verification section, not by a unit test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def sweep_points(convergence_study_module: ModuleType):
    """Three AMG and three POD-2G sweep points at distinct realized coarse dimensions.

    Only ``coarse_dim`` is exercised by the functions under test, so
    ``result`` is left as ``None`` rather than building a full
    ``CGComparisonResult`` fixture; timings are arbitrary fixed values.
    """
    sweep_point = convergence_study_module.SweepPoint
    amg_points = [sweep_point(f"amg-{c}", c, None, 1.0, 2.0) for c in (5, 12, 30)]
    pod_points = [sweep_point(f"pod2g-{c}", c, None, 1.0, 2.0) for c in (8, 14, 40)]
    return amg_points, pod_points


def test_parse_float_list_splits_and_converts(convergence_study_module: ModuleType) -> None:
    assert convergence_study_module._parse_float_list("0.05,0.1,0.2") == [0.05, 0.1, 0.2]


def test_parse_int_list_splits_and_converts(convergence_study_module: ModuleType) -> None:
    assert convergence_study_module._parse_int_list("1,10,100") == [1, 10, 100]


def test_closest_match_picks_nearest_realized_coarse_dimension(
    convergence_study_module: ModuleType, sweep_points: tuple[list, list]
) -> None:
    amg_points, pod_points = sweep_points
    amg_match, pod_match = convergence_study_module._closest_match(amg_points, pod_points)
    assert amg_match.coarse_dim == 12
    assert pod_match.coarse_dim == 14


def test_amortized_seconds_divides_only_setup_cost(convergence_study_module: ModuleType) -> None:
    point = convergence_study_module.SweepPoint(
        "pod2g-test", 10, None, setup_seconds=10.0, solve_seconds=1.0
    )
    assert point.amortized_seconds(1) == pytest.approx(11.0)
    assert point.amortized_seconds(10) == pytest.approx(2.0)


def test_fastest_of_keeps_the_result_from_the_minimum_elapsed_call(
    convergence_study_module: ModuleType,
) -> None:
    fake_durations = iter([0.03, 0.01, 0.02])

    def fake_timed_call() -> tuple[float, str]:
        duration = next(fake_durations)
        return duration, f"result-after-{duration}"

    fastest = convergence_study_module.fastest_of(fake_timed_call, repeats=3)
    elapsed, result = fastest()
    assert elapsed == 0.01
    assert result == "result-after-0.01"
