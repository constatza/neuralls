"""The observation view: one row per planned system, derived from the stored frames only."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import pytest

from neuralls.domain.solver.reporting.observations import (
    OBSERVATION_SCHEMA,
    MethodFrames,
    Outcome,
    observations,
    observations_of_methods,
)
from neuralls.shared.types import PreconditionerName
from tests.observation_support import (
    AMG,
    BROKEN,
    JACOBI,
    PLANNED,
    STOPPING_THRESHOLD_REL,
)

SORT_KEYS = ["method", "dataset", "matrix_id", "rhs_ordinal"]
VALUE_COLUMNS = [
    "outcome",
    "iterations",
    "solve_s",
    "setup_s",
    "planned_rhs",
    "online_s",
    "standalone_s",
    "time_per_iteration_s",
]
AMG_ROWS = {
    # outcome, iterations, solve_s, setup_s, planned_rhs, online_s, standalone_s, time/iteration
    "m1-0": ("converged", 2, 1.0, 6.0, 3, 3.0, 7.0, 0.5),
    "m1-1": ("converged", 5, 2.0, 6.0, 3, 4.0, 8.0, 0.4),
    "m1-2": ("converged", 3, 3.0, 6.0, 3, 5.0, 9.0, 1.0),
    "m2-0": ("converged", 8, 4.0, 12.0, 3, 8.0, 16.0, 0.5),
    "m2-1": ("unconverged", 20, 8.0, 12.0, 3, 12.0, 20.0, 0.4),
    "m2-2": ("solver_error", None, 0.5, 12.0, 3, 4.5, 12.5, None),
    "m3-0": ("setup_failed", None, None, 0.5, 2, None, None, None),
    "m3-1": ("setup_failed", None, None, 0.5, 2, None, None, None),
    "m4-0": ("not_attempted", None, None, None, 1, None, None, None),
}
CORRUPTIONS = (
    "solve_on_failed_setup",
    "unplanned_solve",
    "solve_contradicts_plan",
    "duplicate_solve",
    "foreign_setup_method",
    "foreign_solve_method",
)


@pytest.fixture
def amg_observations(unbalanced_observations: pl.DataFrame) -> pl.DataFrame:
    return unbalanced_observations.filter(pl.col("method") == AMG)


@pytest.fixture
def corrupt_frames(
    unbalanced_frames: Mapping[PreconditionerName, MethodFrames],
) -> Mapping[str, MethodFrames]:
    """``amg`` frames each violating one invariant the view relies on."""
    clean = unbalanced_frames[AMG]
    solve = clean.solve_attempts
    first = solve.head(1)

    def with_solve(frame: pl.DataFrame) -> MethodFrames:
        return MethodFrames(clean.planned_systems, clean.setup_events, frame)

    return {
        "solve_on_failed_setup": with_solve(
            solve.with_columns(
                setup_event_id=pl.when(pl.col("system_id") == "m1-0")
                .then(pl.lit("amg-setup-3"))
                .otherwise(pl.col("setup_event_id"))
            )
        ),
        "unplanned_solve": with_solve(
            solve.with_columns(
                system_id=pl.when(pl.col("system_id") == "m1-0")
                .then(pl.lit("ghost"))
                .otherwise(pl.col("system_id"))
            )
        ),
        "solve_contradicts_plan": with_solve(
            solve.with_columns(
                rhs_ordinal=pl.when(pl.col("system_id") == "m1-0")
                .then(7)
                .otherwise(pl.col("rhs_ordinal"))
            )
        ),
        "duplicate_solve": with_solve(pl.concat([solve, first])),
        "foreign_setup_method": MethodFrames(
            clean.planned_systems,
            clean.setup_events.with_columns(method=pl.lit("other")),
            solve,
        ),
        "foreign_solve_method": with_solve(solve.with_columns(method=pl.lit("other"))),
    }


def _values(frame: pl.DataFrame) -> dict[str, tuple[object, ...]]:
    return {
        row["system_id"]: tuple(row[column] for column in VALUE_COLUMNS)
        for row in frame.iter_rows(named=True)
    }


def test_the_view_has_the_declared_schema(unbalanced_observations: pl.DataFrame) -> None:
    assert dict(unbalanced_observations.schema) == OBSERVATION_SCHEMA


def test_rows_are_sorted_by_method_dataset_matrix_and_ordinal(
    unbalanced_observations: pl.DataFrame,
) -> None:
    assert unbalanced_observations.equals(unbalanced_observations.sort(SORT_KEYS))
    assert unbalanced_observations["method"].unique(maintain_order=True).to_list() == [
        AMG,
        BROKEN,
        JACOBI,
    ]


def test_there_is_one_row_per_planned_system_and_method(
    unbalanced_observations: pl.DataFrame,
) -> None:
    assert unbalanced_observations.height == 3 * len(PLANNED)
    assert unbalanced_observations.group_by("method").len()["len"].to_list() == [len(PLANNED)] * 3


def test_each_system_of_the_unbalanced_workload_has_its_hand_computed_values(
    amg_observations: pl.DataFrame,
) -> None:
    values = _values(amg_observations)

    assert set(values) == set(AMG_ROWS)
    for system, expected in AMG_ROWS.items():
        assert values[system] == pytest.approx(expected), system


def test_the_outcome_values_are_the_outcome_vocabulary(
    unbalanced_observations: pl.DataFrame,
) -> None:
    assert set(unbalanced_observations["outcome"]) <= {outcome.value for outcome in Outcome}


def test_a_method_that_never_got_past_setup_has_only_setup_failed_and_not_attempted_rows(
    unbalanced_observations: pl.DataFrame,
) -> None:
    broken = unbalanced_observations.filter(pl.col("method") == BROKEN)

    assert broken["outcome"].to_list() == ["setup_failed"] * 8 + ["not_attempted"]
    assert broken["solve_s"].null_count() == broken.height


# solve time plus the matrix setup over its planned RHS: amg setups 6.0 (matrix 1) and 12.0
# (matrix 2) over 3 RHS each, jacobi 1.0, 2.0, 0.1 and 0.2 over 3, 3, 2 and 1 RHS
ONLINE_SECONDS: dict[PreconditionerName, dict[str, float]] = {
    AMG: {"m1-0": 1.0 + 2.0, "m1-1": 2.0 + 2.0, "m1-2": 3.0 + 2.0, "m2-0": 4.0 + 4.0},
    JACOBI: {
        "m1-0": 0.5 + 1 / 3,
        "m1-1": 0.6 + 1 / 3,
        "m1-2": 0.7 + 1 / 3,
        "m2-0": 0.8 + 2 / 3,
        "m2-1": 0.9 + 2 / 3,
        "m2-2": 1.0 + 2 / 3,
        "m3-0": 0.1 + 0.1 / 2,
        "m3-1": 0.2 + 0.1 / 2,
        "m4-0": 0.3 + 0.2,
    },
}


@pytest.mark.parametrize("method", [AMG, JACOBI])
def test_online_time_is_solve_time_plus_the_setup_share_of_the_planned_rhs(
    unbalanced_observations: pl.DataFrame, method: PreconditionerName
) -> None:
    observed = {
        row["system_id"]: row["online_s"]
        for row in unbalanced_observations.filter(
            (pl.col("method") == method) & (pl.col("outcome") == "converged")
        ).iter_rows(named=True)
    }

    assert observed == pytest.approx(ONLINE_SECONDS[method])


def test_residual_threshold_and_timing_spread_follow_the_solve_rows(
    amg_observations: pl.DataFrame,
) -> None:
    rows = {row["system_id"]: row for row in amg_observations.iter_rows(named=True)}

    assert rows["m1-0"]["true_residual_rel"] == pytest.approx(STOPPING_THRESHOLD_REL / 2)
    assert rows["m2-1"]["true_residual_rel"] == pytest.approx(1e-3)
    assert rows["m1-0"]["stopping_threshold_rel"] == pytest.approx(STOPPING_THRESHOLD_REL)
    assert rows["m2-1"]["stopping_threshold_rel"] == pytest.approx(STOPPING_THRESHOLD_REL)
    # (1.2 - 0.9) / median 1.0 for three repeats, 0 for a single sample
    assert rows["m1-0"]["timing_spread"] == pytest.approx(0.3)
    assert rows["m1-1"]["timing_spread"] == 0.0
    for system in ("m2-2", "m3-0", "m4-0"):
        assert rows[system]["true_residual_rel"] is None
        assert rows[system]["stopping_threshold_rel"] is None
    for system in ("m3-0", "m4-0"):
        assert rows[system]["timing_spread"] is None


def test_no_methods_give_an_empty_view_with_the_declared_schema() -> None:
    empty = observations_of_methods({})

    assert empty.height == 0
    assert dict(empty.schema) == OBSERVATION_SCHEMA


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_frames_that_violate_a_link_invariant_are_rejected(
    corrupt_frames: Mapping[str, MethodFrames], corruption: str
) -> None:
    frames = corrupt_frames[corruption]

    with pytest.raises(ValueError):
        observations(AMG, frames.planned_systems, frames.setup_events, frames.solve_attempts)
