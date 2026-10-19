"""Tests for normalized benchmark-event frames and workload summaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import polars as pl
import pytest

from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
    UnknownPeakMemory,
)
from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
)
from neuralls.domain.solver.reporting.aggregation import (
    build_matrix_bundle_frame,
    build_system_cost_frame,
    ranking_costs,
    summarize_method_workload,
)
from neuralls.domain.solver.reporting.event_frames import build_setup_frame, build_solve_frame
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId


@pytest.fixture
def method() -> PreconditionerName:
    """The method whose canonical workload is summarized."""
    return PreconditionerName("amg")


@pytest.fixture
def workload_peak_memory() -> MeasuredPeakMemory:
    """A method-workload peak measured across the full execution lifetime."""
    return MeasuredPeakMemory(bytes=500)


@pytest.fixture
def unknown_workload_peak_memory() -> UnknownPeakMemory:
    """An explicitly unavailable method-workload peak."""
    return UnknownPeakMemory()


@pytest.fixture
def setup_frame_schema() -> pl.Schema:
    """Hand-specified normalized setup-event schema."""
    return pl.Schema(
        {
            "setup_event_id": pl.String,
            "dataset": pl.String,
            "matrix_id": pl.Int64,
            "method": pl.String,
            "setup_s": pl.Float64,
            "samples_s": pl.List(pl.Float64),
            "peak_memory_bytes": pl.Int64,
            "memory_status": pl.String,
            "setup_status": pl.String,
            "failure_reason": pl.String,
        }
    )


@pytest.fixture
def solve_frame_schema() -> pl.Schema:
    """Hand-specified normalized solve-attempt schema."""
    return pl.Schema(
        {
            "setup_event_id": pl.String,
            "system_id": pl.String,
            "dataset": pl.String,
            "recipe": pl.String,
            "matrix_id": pl.Int64,
            "rhs_ordinal": pl.Int64,
            "method": pl.String,
            "solve_s": pl.Float64,
            "samples_s": pl.List(pl.Float64),
            "peak_memory_bytes": pl.Int64,
            "memory_status": pl.String,
            "solve_status": pl.String,
            "iterations": pl.Int64,
            "final_relative_residual": pl.Float64,
            "failure_reason": pl.String,
        }
    )


@pytest.fixture
def planned_systems() -> tuple[SystemTag, ...]:
    """Two matrices with three planned RHS systems each and mixed recipes."""
    return (
        SystemTag(DatasetId("ds"), "gaussian", MatrixId(10), 0, SystemId("10-0")),
        SystemTag(DatasetId("ds"), "sinusoidal", MatrixId(10), 1, SystemId("10-1")),
        SystemTag(DatasetId("ds"), "sinusoidal", MatrixId(10), 2, SystemId("10-2")),
        SystemTag(DatasetId("ds"), "gaussian", MatrixId(20), 0, SystemId("20-0")),
        SystemTag(DatasetId("ds"), "gaussian", MatrixId(20), 1, SystemId("20-1")),
        SystemTag(DatasetId("ds"), "gaussian", MatrixId(20), 2, SystemId("20-2")),
    )


@pytest.fixture
def setup_events(method: PreconditionerName) -> tuple[MatrixSetupEvent, ...]:
    """One setup event for each matrix, costing six and twelve seconds."""
    return (
        MatrixSetupEvent(
            event_id=SetupEventId("setup-10"),
            dataset=DatasetId("ds"),
            matrix_id=MatrixId(10),
            method=method,
            setup_s=6.0,
            samples_s=(5.5, 6.0, 6.5),
            peak_memory=MeasuredPeakMemory(bytes=100),
            outcome=SetupCompleted(),
        ),
        MatrixSetupEvent(
            event_id=SetupEventId("setup-20"),
            dataset=DatasetId("ds"),
            matrix_id=MatrixId(20),
            method=method,
            setup_s=12.0,
            samples_s=(11.0, 12.0, 13.0),
            peak_memory=UnknownPeakMemory(),
            outcome=SetupCompleted(),
        ),
    )


@pytest.fixture
def solve_attempts(
    method: PreconditionerName, planned_systems: tuple[SystemTag, ...]
) -> tuple[SolveAttempt, ...]:
    """Six converged attempts costing one through six seconds."""
    setup_ids = (SetupEventId("setup-10"),) * 3 + (SetupEventId("setup-20"),) * 3
    return tuple(
        SolveAttempt(
            setup_event_id=setup_id,
            system_id=tag.system_id,
            dataset=tag.dataset,
            recipe=tag.recipe,
            matrix_id=tag.matrix_id,
            rhs_ordinal=tag.rhs_ordinal,
            method=method,
            solve_s=float(index),
            samples_s=(float(index),),
            peak_memory=MeasuredPeakMemory(bytes=index * 10),
            outcome=SolveConverged(iterations=index, final_relative_residual=1e-9),
        )
        for index, (tag, setup_id) in enumerate(
            zip(planned_systems, setup_ids, strict=True), start=1
        )
    )


@pytest.fixture
def incomplete_attempts(solve_attempts: tuple[SolveAttempt, ...]) -> tuple[SolveAttempt, ...]:
    """Four attempts, one failed, leaving two planned systems unattempted."""
    return (
        solve_attempts[0],
        solve_attempts[1],
        replace(
            solve_attempts[2],
            outcome=SolveFailed(iterations=30, reason="iteration limit"),
        ),
        solve_attempts[3],
    )


@pytest.fixture
def extreme_planned_systems() -> tuple[SystemTag, ...]:
    """Four planned RHS systems on one matrix and two on another."""
    return (
        SystemTag(DatasetId("extreme"), "r", MatrixId(10), 0, SystemId("a-0")),
        SystemTag(DatasetId("extreme"), "r", MatrixId(10), 1, SystemId("a-1")),
        SystemTag(DatasetId("extreme"), "r", MatrixId(10), 2, SystemId("a-2")),
        SystemTag(DatasetId("extreme"), "r", MatrixId(10), 3, SystemId("a-3")),
        SystemTag(DatasetId("extreme"), "r", MatrixId(20), 0, SystemId("b-0")),
        SystemTag(DatasetId("extreme"), "r", MatrixId(20), 1, SystemId("b-1")),
    )


@pytest.fixture
def extreme_setup_events(method: PreconditionerName) -> tuple[MatrixSetupEvent, ...]:
    """Measured setup costs combined with extreme solve magnitudes."""
    return (
        MatrixSetupEvent(
            event_id=SetupEventId("extreme-10"),
            dataset=DatasetId("extreme"),
            matrix_id=MatrixId(10),
            method=method,
            setup_s=6.0,
            samples_s=(6.0,),
            peak_memory=UnknownPeakMemory(),
            outcome=SetupCompleted(),
        ),
        MatrixSetupEvent(
            event_id=SetupEventId("extreme-20"),
            dataset=DatasetId("extreme"),
            matrix_id=MatrixId(20),
            method=method,
            setup_s=12.0,
            samples_s=(12.0,),
            peak_memory=UnknownPeakMemory(),
            outcome=SetupCompleted(),
        ),
    )


@pytest.fixture
def reordered_extreme_attempts(
    method: PreconditionerName,
    extreme_planned_systems: tuple[SystemTag, ...],
) -> tuple[SolveAttempt, ...]:
    """Interleaved matrix events whose flat and grouped IEEE-754 sums differ."""
    event_order = (0, 1, 2, 4, 5, 3)
    solve_times = (1e16, 1.0, 1.0, 1.0, 1.0, 1e16)
    return tuple(
        SolveAttempt(
            setup_event_id=(
                SetupEventId("extreme-10")
                if extreme_planned_systems[index].matrix_id == MatrixId(10)
                else SetupEventId("extreme-20")
            ),
            system_id=extreme_planned_systems[index].system_id,
            dataset=extreme_planned_systems[index].dataset,
            recipe=extreme_planned_systems[index].recipe,
            matrix_id=extreme_planned_systems[index].matrix_id,
            rhs_ordinal=extreme_planned_systems[index].rhs_ordinal,
            method=method,
            solve_s=solve_s,
            samples_s=(solve_s,),
            peak_memory=UnknownPeakMemory(),
            outcome=SolveConverged(iterations=1, final_relative_residual=0.0),
        )
        for index, solve_s in zip(event_order, solve_times, strict=True)
    )


@pytest.fixture
def invalid_link_attempts(solve_attempts: tuple[SolveAttempt, ...]) -> tuple[SolveAttempt, ...]:
    """Attempts whose setup link is unknown or disagrees on a linkage key."""
    return (
        replace(solve_attempts[0], setup_event_id=SetupEventId("missing")),
        replace(solve_attempts[0], dataset=DatasetId("other")),
        replace(solve_attempts[0], matrix_id=MatrixId(20)),
        replace(solve_attempts[0], method=PreconditionerName("jacobi")),
    )


@pytest.fixture
def no_attempts() -> tuple[SolveAttempt, ...]:
    """A method workload interrupted before its first solve attempt."""
    return ()


@pytest.fixture
def failed_setup_events(
    setup_events: tuple[MatrixSetupEvent, ...],
) -> tuple[MatrixSetupEvent, ...]:
    """A setup event that failed before any linked solve could be attempted."""
    return (
        replace(setup_events[0], outcome=SetupFailed(reason="factorization failed")),
        setup_events[1],
    )


@pytest.fixture
def duplicate_missing_summary_factory(
    method: PreconditionerName,
    unknown_workload_peak_memory: UnknownPeakMemory,
) -> Callable[[], IncompleteWorkloadSummary]:
    """Construct an incomplete summary that repeats one missing system identity."""

    def construct() -> IncompleteWorkloadSummary:
        return IncompleteWorkloadSummary(
            method=method,
            n_planned_rhs=2,
            n_attempted_rhs=0,
            n_converged=0,
            setup_total_s=0.0,
            solve_total_s=0.0,
            workload_online_s=0.0,
            online_s_per_attempted_rhs=None,
            workload_peak_memory=unknown_workload_peak_memory,
            missing_system_ids=(SystemId("missing"), SystemId("missing")),
        )

    return construct


@pytest.fixture
def inconsistent_complete_summary_factory(
    method: PreconditionerName,
    unknown_workload_peak_memory: UnknownPeakMemory,
) -> Callable[[], CompleteWorkloadSummary]:
    """Construct a summary whose derived total is materially inconsistent."""

    def construct() -> CompleteWorkloadSummary:
        return CompleteWorkloadSummary(
            method=method,
            n_planned_rhs=1,
            n_attempted_rhs=1,
            n_converged=1,
            setup_total_s=1.0,
            solve_total_s=2.0,
            workload_online_s=3.01,
            online_s_per_attempted_rhs=3.01,
            workload_peak_memory=unknown_workload_peak_memory,
        )

    return construct


def test_event_frames_normalize_variant_payloads(
    setup_events: tuple[MatrixSetupEvent, ...],
    solve_attempts: tuple[SolveAttempt, ...],
    setup_frame_schema: pl.Schema,
    solve_frame_schema: pl.Schema,
) -> None:
    """A missing event column or variant projection breaks canonical artifact tables."""
    setups = build_setup_frame(setup_events)
    solves = build_solve_frame(solve_attempts)

    assert setups.height == 2
    assert setups.schema == setup_frame_schema
    assert solves.height == 6
    assert solves.schema == solve_frame_schema
    assert setups["memory_status"].to_list() == ["measured", "unknown"]
    assert setups["peak_memory_bytes"].to_list() == [100, None]
    assert solves["solve_status"].to_list() == ["converged"] * 6


def test_empty_event_frames_retain_exact_schemas(
    setup_frame_schema: pl.Schema,
    solve_frame_schema: pl.Schema,
) -> None:
    """Empty artifacts must retain the same dtypes as populated artifacts."""
    assert build_setup_frame(()).schema == setup_frame_schema
    assert build_solve_frame(()).schema == solve_frame_schema


def test_event_frames_preserve_failed_outcome_data(
    failed_setup_events: tuple[MatrixSetupEvent, ...],
    incomplete_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Failure status, reason, and state-specific nulls must survive serialization."""
    setup_row = build_setup_frame(failed_setup_events[:1]).row(0, named=True)
    failed_system_id = incomplete_attempts[2].system_id
    solve_row = (
        build_solve_frame(incomplete_attempts)
        .filter(pl.col("system_id") == failed_system_id)
        .row(0, named=True)
    )

    assert setup_row["setup_status"] == "failed"
    assert setup_row["failure_reason"] == "factorization failed"
    assert solve_row["solve_status"] == "failed"
    assert solve_row["final_relative_residual"] is None
    assert solve_row["failure_reason"] == "iteration limit"


def test_system_costs_allocate_each_matrix_setup_over_planned_rhs(
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    solve_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Counting attempts instead of planned matrix membership misallocates shared setup."""
    costs = build_system_cost_frame(planned_systems, setup_events, solve_attempts).sort(
        ["matrix_id", "rhs_ordinal"]
    )

    assert costs["planned_rhs_count"].to_list() == [3, 3, 3, 3, 3, 3]
    assert costs["allocated_setup_s"].sum() == pytest.approx(18.0)
    assert costs["amortized_online_s"].to_list() == pytest.approx((3, 4, 5, 8, 9, 10))


def test_matrix_bundle_costs_charge_setup_once(
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    solve_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Grouping setup-bearing solve rows would multiply one setup by RHS count."""
    bundles = build_matrix_bundle_frame(planned_systems, setup_events, solve_attempts).sort(
        "matrix_id"
    )

    assert bundles["planned_rhs_count"].to_list() == [3, 3]
    assert bundles["attempted_rhs_count"].to_list() == [3, 3]
    assert bundles["matrix_online_s"].to_list() == pytest.approx((12.0, 27.0))
    assert bundles["setup_s"].sum() == pytest.approx(18.0)


def test_setup_link_is_validated_before_cost_allocation(
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    invalid_link_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Unknown or contradictory setup links must not receive a fabricated allocation."""
    for attempt in invalid_link_attempts:
        with pytest.raises(ValueError, match="setup"):
            build_system_cost_frame(planned_systems, setup_events, (attempt,))


def test_failed_setup_cannot_back_a_solve_attempt(
    planned_systems: tuple[SystemTag, ...],
    failed_setup_events: tuple[MatrixSetupEvent, ...],
    solve_attempts: tuple[SolveAttempt, ...],
) -> None:
    """A failed setup cannot provide valid prepared state for a solve."""
    with pytest.raises(ValueError, match="failed setup"):
        build_system_cost_frame(planned_systems, failed_setup_events, solve_attempts)


def test_matrix_bundle_retains_failed_setup_without_attempts(
    planned_systems: tuple[SystemTag, ...],
    failed_setup_events: tuple[MatrixSetupEvent, ...],
    no_attempts: tuple[SolveAttempt, ...],
) -> None:
    """A failed setup remains a measured matrix cost even when no solve starts."""
    row = build_matrix_bundle_frame(planned_systems, failed_setup_events[:1], no_attempts).row(
        0, named=True
    )

    assert row["setup_s"] == 6.0
    assert row["setup_status"] == "failed"
    assert row["failure_reason"] == "factorization failed"
    assert row["attempted_rhs_count"] == 0
    assert row["solve_total_s"] == 0.0


def test_recipe_groups_share_one_matrix_charge(
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    solve_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Recipe grouping after allocation must not reproduce raw matrix setup rows."""
    recipe_costs = (
        build_system_cost_frame(planned_systems, setup_events[:1], solve_attempts[:3])
        .group_by("recipe")
        .agg(pl.col("allocated_setup_s").sum())
    )

    assert recipe_costs["allocated_setup_s"].sum() == pytest.approx(6.0)


def test_complete_workload_summary_reconciles_raw_events(
    method: PreconditionerName,
    workload_peak_memory: MeasuredPeakMemory,
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    solve_attempts: tuple[SolveAttempt, ...],
) -> None:
    """A complete method summary must derive each total once from canonical events."""
    summary = summarize_method_workload(
        method,
        planned_systems,
        setup_events,
        solve_attempts,
        workload_peak_memory,
    )

    assert isinstance(summary, CompleteWorkloadSummary)
    assert (summary.n_planned_rhs, summary.n_attempted_rhs, summary.n_converged) == (6, 6, 6)
    assert summary.setup_total_s == pytest.approx(18.0)
    assert summary.solve_total_s == pytest.approx(21.0)
    assert summary.workload_online_s == pytest.approx(39.0)
    assert summary.online_s_per_attempted_rhs == pytest.approx(6.5)
    assert ranking_costs((summary,)) == {method: 39.0}


def test_workload_totals_are_derived_from_matrix_bundles(
    method: PreconditionerName,
    workload_peak_memory: MeasuredPeakMemory,
    extreme_planned_systems: tuple[SystemTag, ...],
    extreme_setup_events: tuple[MatrixSetupEvent, ...],
    reordered_extreme_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Event-order rounding must not make workload and matrix reports disagree."""
    assert [attempt.matrix_id for attempt in reordered_extreme_attempts] == [
        MatrixId(10),
        MatrixId(10),
        MatrixId(10),
        MatrixId(20),
        MatrixId(20),
        MatrixId(10),
    ]
    bundles = build_matrix_bundle_frame(
        extreme_planned_systems, extreme_setup_events, reordered_extreme_attempts
    )
    summary = summarize_method_workload(
        method,
        extreme_planned_systems,
        extreme_setup_events,
        reordered_extreme_attempts,
        workload_peak_memory,
    )

    assert summary.setup_total_s == bundles["setup_s"].sum()
    assert summary.solve_total_s == bundles["solve_total_s"].sum()
    assert summary.workload_online_s == bundles["matrix_online_s"].sum()


def test_incomplete_workload_is_visible_and_not_rankable(
    method: PreconditionerName,
    unknown_workload_peak_memory: UnknownPeakMemory,
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    incomplete_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Missing systems must not make an interrupted method look artificially cheap."""
    summary = summarize_method_workload(
        method,
        planned_systems,
        setup_events,
        incomplete_attempts,
        unknown_workload_peak_memory,
    )

    assert isinstance(summary, IncompleteWorkloadSummary)
    assert (summary.n_planned_rhs, summary.n_attempted_rhs, summary.n_converged) == (6, 4, 3)
    assert summary.missing_system_ids == (SystemId("20-1"), SystemId("20-2"))
    assert summary.online_s_per_attempted_rhs == pytest.approx(7.0)
    with pytest.raises(TypeError, match="incomplete"):
        ranking_costs((summary,))


def test_unattempted_workload_has_no_attempted_rhs_average(
    method: PreconditionerName,
    unknown_workload_peak_memory: UnknownPeakMemory,
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    no_attempts: tuple[SolveAttempt, ...],
) -> None:
    """An interrupted workload with no attempts has no per-attempt average."""
    summary = summarize_method_workload(
        method,
        planned_systems,
        setup_events,
        no_attempts,
        unknown_workload_peak_memory,
    )

    assert isinstance(summary, IncompleteWorkloadSummary)
    assert isinstance(summary.solve_total_s, float)
    assert summary.solve_total_s == 0.0
    assert summary.online_s_per_attempted_rhs is None


def test_workload_summaries_are_frozen_and_slotted(
    method: PreconditionerName,
    unknown_workload_peak_memory: UnknownPeakMemory,
    planned_systems: tuple[SystemTag, ...],
    setup_events: tuple[MatrixSetupEvent, ...],
    solve_attempts: tuple[SolveAttempt, ...],
) -> None:
    """Mutable aggregate facts could drift away from their canonical event frames."""
    summary = summarize_method_workload(
        method,
        planned_systems,
        setup_events,
        solve_attempts,
        unknown_workload_peak_memory,
    )

    assert not hasattr(summary, "__dict__")
    with pytest.raises(FrozenInstanceError):
        summary.setup_total_s = 0.0


def test_incomplete_summary_rejects_duplicate_missing_system_ids(
    duplicate_missing_summary_factory: Callable[[], IncompleteWorkloadSummary],
) -> None:
    """Missing membership must identify distinct planned systems."""
    with pytest.raises(ValueError, match="distinct"):
        duplicate_missing_summary_factory()


def test_workload_summary_rejects_material_total_disagreement(
    inconsistent_complete_summary_factory: Callable[[], CompleteWorkloadSummary],
) -> None:
    """The IEEE-754 tolerance must not hide a meaningful timing discrepancy."""
    with pytest.raises(ValueError, match="setup plus solve"):
        inconsistent_complete_summary_factory()
