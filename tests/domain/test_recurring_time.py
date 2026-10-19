"""Recurring versus one-time cost: a shared setup is charged once, not once per system.

Fixture: six systems and two preconditioners. ``neural`` shares its setup
(a checkpoint load of 2.0 s copied into every sibling by the setup cache) and
solves in 0.1 s; ``jacobi`` has no setup and solves in 0.3 s; ``amg`` builds a
per-matrix setup of 0.5 s and solves in 0.2 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.benchmarking.breakeven import break_even
from neuralls.domain.benchmarking.costs import seconds
from neuralls.domain.benchmarking.profiles import performance_profiles
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost import StageCost
from neuralls.domain.solver.models.labels import GlobalLabel
from neuralls.domain.solver.models.outcomes import Solved
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.models.statistics import BreakEvenAt
from neuralls.domain.solver.reporting.aggregation import (
    convergence_by_level,
    outcomes_by_system,
    report_level,
)
from neuralls.domain.solver.reporting.metrics_frame import SystemTag, build_metrics_frame
from neuralls.domain.solver.reporting.report_levels import GlobalLevel
from neuralls.shared.types import CostProvenance, DatasetId, MatrixId

N_SYSTEMS = 6
SHARED_SETUP_S = 2.0
NEURAL_SOLVE_S = 0.1
JACOBI_SOLVE_S = 0.3
AMG_SETUP_S = 0.5
AMG_SOLVE_S = 0.2
GENERATION_S = 4.0

SHARING = {
    "neural": SetupSharingPolicy(share_setup=True, share_generation=False),
    "jacobi": SetupSharingPolicy(share_setup=False, share_generation=False),
    "amg": SetupSharingPolicy(share_setup=False, share_generation=False),
}
LABEL = GlobalLabel(("gauss",))


def _result(name: str, *, setup: float, solve: float) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=5,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 1e-9],
        residual_history_abs=[1.0, 1e-9],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=setup,
        solve_time_seconds=solve,
    )


def _tag(ordinal: int) -> SystemTag:
    return SystemTag(
        dataset=DatasetId("ds"),
        recipe="gauss",
        matrix_id=MatrixId(0),
        rhs_ordinal=ordinal,
        system_id=f"s{ordinal}",
    )


@pytest.fixture
def results() -> dict[SystemTag, dict[str, CGComparisonResult]]:
    return {
        _tag(i): {
            "neural": _result("neural", setup=SHARED_SETUP_S, solve=NEURAL_SOLVE_S),
            "jacobi": _result("jacobi", setup=0.0, solve=JACOBI_SOLVE_S),
            "amg": _result("amg", setup=AMG_SETUP_S, solve=AMG_SOLVE_S),
        }
        for i in range(N_SYSTEMS)
    }


@pytest.fixture
def frame(results):
    return build_metrics_frame(results, SHARING)


def _column(frame, name: str, column: str) -> list[float]:
    return frame.filter(frame["preconditioner"] == name)[column].to_list()


def test_shared_setup_is_excluded_from_recurring_time(frame) -> None:
    assert _column(frame, "neural", "recurring_s") == pytest.approx([NEURAL_SOLVE_S] * N_SYSTEMS)


def test_per_matrix_setup_stays_in_recurring_time(frame) -> None:
    assert _column(frame, "amg", "recurring_s") == pytest.approx(
        [AMG_SETUP_S + AMG_SOLVE_S] * N_SYSTEMS
    )


def test_time_to_solution_keeps_the_full_wall_time(frame) -> None:
    assert _column(frame, "neural", "time_to_solution_s") == pytest.approx(
        [SHARED_SETUP_S + NEURAL_SOLVE_S] * N_SYSTEMS
    )


def test_shared_generation_is_excluded_from_recurring_time() -> None:
    generation = StageCost(
        wall_time_seconds=GENERATION_S, peak_memory_bytes=None, provenance=CostProvenance.HISTORICAL
    )
    result = _result("neural", setup=SHARED_SETUP_S, solve=NEURAL_SOLVE_S)
    result.generation_cost = generation
    policy = {"neural": SetupSharingPolicy(share_setup=True, share_generation=True)}

    frame = build_metrics_frame({_tag(0): {"neural": result}}, policy)

    assert frame["recurring_s"].to_list() == pytest.approx([NEURAL_SOLVE_S])
    assert frame["time_to_solution_s"].to_list() == pytest.approx(
        [GENERATION_S + SHARED_SETUP_S + NEURAL_SOLVE_S]
    )


def test_missing_sharing_policy_raises(results) -> None:
    with pytest.raises(KeyError):
        build_metrics_frame(results, {"neural": SHARING["neural"]})


def test_outcomes_carry_the_recurring_time(frame) -> None:
    neural = outcomes_by_system(frame, GlobalLevel())[LABEL]["neural"]

    assert len(neural) == N_SYSTEMS
    assert all(isinstance(o, Solved) for o in neural.values())
    assert [o.recurring_time_s for o in neural.values() if isinstance(o, Solved)] == (
        pytest.approx([NEURAL_SOLVE_S] * N_SYSTEMS)
    )


def test_performance_profile_ranks_the_shared_setup_method_best_everywhere(frame) -> None:
    profiles = performance_profiles(outcomes_by_system(frame, GlobalLevel())[LABEL], seconds)

    # Ratio 1.0 on every system for the best method; the others never reach it.
    assert profiles["neural"].solved_at == pytest.approx([1.0] * N_SYSTEMS)
    assert min(profiles["jacobi"].solved_at) == pytest.approx(JACOBI_SOLVE_S / NEURAL_SOLVE_S)
    assert min(profiles["amg"].solved_at) == pytest.approx(
        (AMG_SETUP_S + AMG_SOLVE_S) / NEURAL_SOLVE_S
    )


def test_ert_charges_the_shared_setup_once(frame) -> None:
    summary = convergence_by_level(frame, GlobalLevel(), SHARING)[LABEL]

    assert summary["neural"].ert_seconds == pytest.approx(
        (SHARED_SETUP_S + N_SYSTEMS * NEURAL_SOLVE_S) / N_SYSTEMS
    )
    assert summary["amg"].ert_seconds == pytest.approx(AMG_SETUP_S + AMG_SOLVE_S)


def test_ert_agrees_with_the_projection(frame) -> None:
    projected = report_level(frame, GlobalLevel(), SHARING)[LABEL]["neural"]
    ert = convergence_by_level(frame, GlobalLevel(), SHARING)[LABEL]["neural"].ert_seconds

    assert projected.one_time_seconds == pytest.approx(SHARED_SETUP_S)
    assert ert == pytest.approx(
        projected.per_solved_system_seconds + projected.one_time_seconds / N_SYSTEMS
    )


def test_break_even_charges_the_shared_setup_once(frame) -> None:
    projected = report_level(frame, GlobalLevel(), SHARING)[LABEL]

    # 2.0 + 0.1 n = 0.3 n  ->  n = 10.
    result = break_even(projected["neural"], projected["jacobi"])
    assert isinstance(result, BreakEvenAt)
    assert result.n_systems == pytest.approx(10.0)
