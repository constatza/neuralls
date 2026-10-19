"""Matrix-first event aggregation and legacy projection reporting."""

from __future__ import annotations

import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from neuralls.domain.benchmarking.uncertainty import ert_interval
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    PeakMemory,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.labels import GroupLabel
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.models.statistics import IterationStats
from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
    MethodWorkloadSummary,
)
from neuralls.domain.solver.reporting.event_frames import build_setup_frame, build_solve_frame
from neuralls.domain.solver.reporting.metrics_frame import (
    PRECONDITIONER,
    SYSTEM_ID,
)
from neuralls.domain.solver.reporting.report_levels import (
    ReportLevel,
    group_label,
    level_columns,
    mixtures_by_key,
)
from neuralls.shared.constants import DEFAULT_RANDOM_SEED
from neuralls.shared.types import MatrixId, PreconditionerName, SetupEventId, SystemId

_PLANNED_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "dataset": pl.String,
    "matrix_id": pl.Int64,
    SYSTEM_ID: pl.String,
}


def build_system_cost_frame(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
) -> pl.DataFrame:
    """Build attempted-system costs with setup allocated by planned matrix membership."""
    _validate_event_links(planned, setup_events, attempts)
    planned_counts = _planned_matrix_counts(planned)
    setup_costs = build_setup_frame(setup_events).select("setup_event_id", "setup_s")
    return (
        build_solve_frame(attempts)
        .join(setup_costs, on="setup_event_id", how="left", validate="m:1")
        .join(planned_counts, on=["dataset", "matrix_id"], how="left", validate="m:1")
        .with_columns(
            allocated_setup_s=pl.col("setup_s") / pl.col("planned_rhs_count"),
        )
        .with_columns(
            amortized_online_s=pl.col("solve_s") + pl.col("allocated_setup_s"),
        )
    )


def build_matrix_bundle_frame(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
) -> pl.DataFrame:
    """Build one setup-aware online-cost row per method and matrix."""
    _validate_event_links(planned, setup_events, attempts)
    planned_counts = _planned_matrix_counts(planned)
    solve_totals = (
        build_solve_frame(attempts)
        .group_by("setup_event_id")
        .agg(
            attempted_rhs_count=pl.len(),
            converged_rhs_count=(pl.col("solve_status") == "converged").sum(),
            solve_total_s=pl.col("solve_s").sum(),
        )
    )
    return (
        build_setup_frame(setup_events)
        .join(planned_counts, on=["dataset", "matrix_id"], how="left", validate="m:1")
        .join(solve_totals, on="setup_event_id", how="left", validate="1:1")
        .with_columns(
            pl.col("attempted_rhs_count").fill_null(0),
            pl.col("converged_rhs_count").fill_null(0),
            pl.col("solve_total_s").fill_null(0.0),
        )
        .with_columns(matrix_online_s=pl.col("setup_s") + pl.col("solve_total_s"))
    )


def summarize_method_workload(
    method: PreconditionerName,
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    workload_peak_memory: PeakMemory,
) -> MethodWorkloadSummary:
    """Derive one complete or explicitly incomplete summary for ``method``."""
    method_setups = tuple(event for event in setup_events if event.method == method)
    method_attempts = tuple(attempt for attempt in attempts if attempt.method == method)
    bundles = build_matrix_bundle_frame(planned, method_setups, method_attempts)
    attempted_ids = {attempt.system_id for attempt in method_attempts}
    missing_ids = tuple(tag.system_id for tag in planned if tag.system_id not in attempted_ids)
    setup_total_s = float(bundles["setup_s"].sum() or 0.0)
    solve_total_s = float(bundles["solve_total_s"].sum() or 0.0)
    workload_online_s = float(bundles["matrix_online_s"].sum() or 0.0)
    n_attempted_rhs = len(method_attempts)
    n_converged = sum(isinstance(attempt.outcome, SolveConverged) for attempt in method_attempts)
    online_s_per_attempted_rhs = (
        workload_online_s / n_attempted_rhs if n_attempted_rhs > 0 else None
    )
    if missing_ids:
        return IncompleteWorkloadSummary(
            method=method,
            n_planned_rhs=len(planned),
            n_attempted_rhs=n_attempted_rhs,
            n_converged=n_converged,
            setup_total_s=setup_total_s,
            solve_total_s=solve_total_s,
            workload_online_s=workload_online_s,
            online_s_per_attempted_rhs=online_s_per_attempted_rhs,
            workload_peak_memory=workload_peak_memory,
            missing_system_ids=missing_ids,
        )
    return CompleteWorkloadSummary(
        method=method,
        n_planned_rhs=len(planned),
        n_attempted_rhs=n_attempted_rhs,
        n_converged=n_converged,
        setup_total_s=setup_total_s,
        solve_total_s=solve_total_s,
        workload_online_s=workload_online_s,
        online_s_per_attempted_rhs=online_s_per_attempted_rhs,
        workload_peak_memory=workload_peak_memory,
    )


def ranking_costs(
    summaries: Sequence[MethodWorkloadSummary],
) -> Mapping[PreconditionerName, float]:
    """Return rankable online workload costs, rejecting incomplete methods."""
    costs: dict[PreconditionerName, float] = {}
    for summary in summaries:
        if isinstance(summary, IncompleteWorkloadSummary):
            raise TypeError(f"incomplete workload for method {summary.method!r} is not rankable")
        if summary.method in costs:
            raise ValueError(f"duplicate workload summary for method {summary.method!r}")
        costs[summary.method] = summary.workload_online_s
    return costs


def _planned_matrix_counts(planned: Sequence[SystemTag]) -> pl.DataFrame:
    """Count complete planned membership before any solve-attempt join."""
    rows = [
        {"dataset": tag.dataset, "matrix_id": tag.matrix_id, SYSTEM_ID: tag.system_id}
        for tag in planned
    ]
    return (
        pl.DataFrame(rows, schema=_PLANNED_SCHEMA)
        .group_by("dataset", "matrix_id", maintain_order=True)
        .agg(pl.len().alias("planned_rhs_count"))
    )


def _validate_event_links(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
) -> None:
    """Reject ambiguous identities and broken setup links before allocating costs."""
    planned_by_id: dict[SystemId, SystemTag] = {}
    planned_matrices: set[tuple[str, MatrixId]] = set()
    for tag in planned:
        if tag.system_id in planned_by_id:
            raise ValueError(f"duplicate planned system ID: {tag.system_id!r}")
        planned_by_id[tag.system_id] = tag
        planned_matrices.add((tag.dataset, tag.matrix_id))

    setup_by_id: dict[SetupEventId, MatrixSetupEvent] = {}
    setup_keys: set[tuple[str, MatrixId, PreconditionerName]] = set()
    for event in setup_events:
        if event.event_id in setup_by_id:
            raise ValueError(f"duplicate setup event ID: {event.event_id!r}")
        setup_key = (event.dataset, event.matrix_id, event.method)
        if setup_key in setup_keys:
            raise ValueError("duplicate setup event for one dataset, matrix, and method")
        if (event.dataset, event.matrix_id) not in planned_matrices:
            raise ValueError("setup event does not belong to a planned matrix")
        setup_by_id[event.event_id] = event
        setup_keys.add(setup_key)

    attempt_keys: set[tuple[SystemId, PreconditionerName]] = set()
    for attempt in attempts:
        setup = setup_by_id.get(attempt.setup_event_id)
        if setup is None:
            raise ValueError(f"unknown setup event ID: {attempt.setup_event_id!r}")
        if isinstance(setup.outcome, SetupFailed):
            # The type is valid; its failed state makes the relationship invalid.
            raise ValueError("solve attempt cannot link to a failed setup event")  # noqa: TRY004
        if (attempt.dataset, attempt.matrix_id, attempt.method) != (
            setup.dataset,
            setup.matrix_id,
            setup.method,
        ):
            raise ValueError("solve attempt does not match its linked setup event")
        planned_tag = planned_by_id.get(attempt.system_id)
        if planned_tag is None:
            raise ValueError(f"solve attempt is not a planned system: {attempt.system_id!r}")
        if (
            attempt.dataset,
            attempt.recipe,
            attempt.matrix_id,
            attempt.rhs_ordinal,
        ) != (
            planned_tag.dataset,
            planned_tag.recipe,
            planned_tag.matrix_id,
            planned_tag.rhs_ordinal,
        ):
            raise ValueError("solve attempt does not match its planned system tag")
        attempt_key = (attempt.system_id, attempt.method)
        if attempt_key in attempt_keys:
            raise ValueError("duplicate solve attempt for one system and method")
        attempt_keys.add(attempt_key)


def report_level(
    frame: pl.DataFrame,
    level: ReportLevel,
    sharing_by_name: Mapping[str, SetupSharingPolicy],
) -> dict[GroupLabel, dict[str, ProjectedComparisonResult]]:
    """Reduce the frame to one projection per (group, preconditioner).

    Solve time and iterations average over every run, a run stopped by the
    iteration cap included at its measured cost, so the projection is a cost
    per solved system (`ProjectedComparisonResult`); their spreads cover the
    converged runs only. A preconditioner that converged on no system of a
    group has nothing to project and is absent from that group;
    `convergence_by_level` still reports it.

    Args:
        frame: Output of `build_metrics_frame`.
        level: Which columns, besides the preconditioner, key the groups.
        sharing_by_name: Each preconditioner's setup/generation sharing policy.

    Returns:
        Projections per group label, groups and preconditioners in first-seen order.

    Raises:
        KeyError: If a preconditioner in the frame has no sharing policy.
    """
    aggregated = _aggregate_by_preconditioner(frame, level_columns(level))
    mixtures = mixtures_by_key(frame, level_columns(level))
    report: dict[GroupLabel, dict[str, ProjectedComparisonResult]] = {}
    for row in aggregated.iter_rows(named=True):
        agg = _Aggregate.from_row(row)
        if agg.n_converged == 0:
            continue
        report.setdefault(group_label(level, row, mixtures), {})[agg.name] = _projection(
            agg, sharing_by_name[agg.name]
        )
    return report


def _aggregate_by_preconditioner(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per (group, preconditioner): costs over all runs, spreads over converged ones.

    Setup, generation, solve time and iterations are paid whether or not the
    solve converged, so their totals cover every run (a failed run at its
    measured cost). The spreads `solve_std` and `iterations_std` describe a
    successful solve and so cover the converged runs only.
    """
    converged = pl.col("converged")
    return frame.group_by([*keys, PRECONDITIONER], maintain_order=True).agg(
        n_systems=pl.len(),
        n_converged=converged.sum(),
        setup_total=pl.col("setup_s").sum(),
        generation_mean=pl.col("generation_s").mean(),
        solve_mean=pl.col("solve_s").mean(),
        solve_std=pl.col("solve_s").filter(converged).std(),
        iterations_total=pl.col("iterations").sum(),
        iterations_std=pl.col("iterations").filter(converged).std(),
    )


@dataclass(frozen=True, slots=True)
class _Aggregate:
    """Typed form of one `_aggregate_by_preconditioner` row, parsed once at the polars boundary."""

    name: str
    n_systems: int
    n_converged: int
    setup_total: float
    generation_mean: float | None
    solve_mean: float | None
    solve_std: float | None
    iterations_total: int
    iterations_std: float | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> _Aggregate:
        return cls(
            name=row[PRECONDITIONER],
            n_systems=row["n_systems"],
            n_converged=row["n_converged"],
            setup_total=row["setup_total"],
            generation_mean=row["generation_mean"],
            solve_mean=row["solve_mean"],
            solve_std=row["solve_std"],
            iterations_total=row["iterations_total"],
            iterations_std=row["iterations_std"],
        )


def _projection(agg: _Aggregate, sharing: SetupSharingPolicy) -> ProjectedComparisonResult:
    """One preconditioner's projection (needs at least one converged run)."""
    n = agg.n_systems
    # Means scaled to n, so the per-system average (total / n) is that mean
    # rather than diluted by unavailable results (generation or an unmeasured solve).
    return ProjectedComparisonResult(
        preconditioner_name=agg.name,
        n_systems=n,
        n_converged=agg.n_converged,
        setup_sharing=sharing,
        setup_cost_total_seconds=agg.setup_total,
        generation_cost_total_seconds=(
            None if agg.generation_mean is None else agg.generation_mean * n
        ),
        total_solve_time_seconds=0.0 if agg.solve_mean is None else agg.solve_mean * n,
        total_iterations=agg.iterations_total,
        solve_time_std_seconds=agg.solve_std,
        iterations_std=agg.iterations_std,
    )


def convergence_by_level(
    frame: pl.DataFrame,
    level: ReportLevel,
    sharing_by_name: Mapping[str, SetupSharingPolicy],
) -> dict[GroupLabel, dict[str, ConvergenceSummary]]:
    """Each group's per-preconditioner success count and expected running time.

    Unlike `report_level`, every preconditioner of a group appears, including
    those that never converged. ERT (COCO) charges failed runs their measured
    cost: the one-time shared cost (see `_one_time_seconds`, the definition
    `ProjectedComparisonResult.one_time_seconds` uses) plus the recurring time
    of all runs, divided by the number of successes. It is `None` when nothing
    converged or when any run's recurring time is incomparable (an unavailable
    cost would silently understate it). Also carries the iteration statistics
    of the converged runs and, when ERT is defined, its bootstrap interval.

    Raises:
        KeyError: If a preconditioner in the frame has no sharing policy.
    """
    keys = level_columns(level)
    summary = frame.group_by([*keys, PRECONDITIONER], maintain_order=True).agg(
        n_systems=pl.len(),
        n_converged=pl.col("converged").sum(),
        recurring_total=pl.col("recurring_s").sum(),
        n_incomparable=pl.col("recurring_s").null_count(),
        times=pl.col("recurring_s"),
        flags=pl.col("converged"),
        setup_mean=pl.col("setup_s").mean(),
        generation_mean=pl.col("generation_s").mean(),
        converged_iterations=pl.col("iterations").filter(pl.col("converged")),
    )
    mixtures = mixtures_by_key(frame, keys)
    out: dict[GroupLabel, dict[str, ConvergenceSummary]] = {}
    for row in summary.iter_rows(named=True):
        name = row[PRECONDITIONER]
        out.setdefault(group_label(level, row, mixtures), {})[name] = _convergence_summary(
            row, sharing_by_name[name]
        )
    return out


def _convergence_summary(row: Mapping[str, Any], sharing: SetupSharingPolicy) -> ConvergenceSummary:
    """One preconditioner's success count, ERT, ERT interval and iteration statistics.

    ERT = (one-time cost + recurring time of all runs) / successes, defined only
    when something converged and every run's recurring time is comparable.
    """
    n_converged = row["n_converged"]
    one_time = sharing.one_time_seconds(row["setup_mean"], row["generation_mean"])
    measurable = n_converged > 0 and row["n_incomparable"] == 0
    return ConvergenceSummary(
        n_systems=row["n_systems"],
        n_converged=n_converged,
        ert_seconds=(one_time + row["recurring_total"]) / n_converged if measurable else None,
        iterations=_iteration_stats(row["converged_iterations"]),
        ert_interval=(
            ert_interval(
                row["times"], row["flags"], one_time_seconds=one_time, seed=DEFAULT_RANDOM_SEED
            )
            if measurable
            else None
        ),
    )


def _iteration_stats(converged_iterations: Sequence[int]) -> IterationStats | None:
    """Mean, median and sample std (ddof=1) of the converged runs' iterations.

    `None` when nothing converged; the std is `None` for a single run.
    """
    if not converged_iterations:
        return None
    std = statistics.stdev(converged_iterations) if len(converged_iterations) > 1 else None
    return IterationStats(
        mean=statistics.fmean(converged_iterations),
        median=statistics.median(converged_iterations),
        std=std,
    )


def outcomes_by_system(
    frame: pl.DataFrame, level: ReportLevel
) -> dict[GroupLabel, dict[str, dict[str, SystemOutcome]]]:
    """Each group's per-preconditioner, per-system `Solved` or `Failed`, in frame order."""
    mixtures = mixtures_by_key(frame, level_columns(level))
    out: dict[GroupLabel, dict[str, dict[str, SystemOutcome]]] = {}
    for row in frame.iter_rows(named=True):
        outcome: SystemOutcome = (
            Solved(iterations=row["iterations"], recurring_time_s=row["recurring_s"])
            if row["converged"]
            else Failed()
        )
        by_name = out.setdefault(group_label(level, row, mixtures), {})
        by_name.setdefault(row[PRECONDITIONER], {})[row[SYSTEM_ID]] = outcome
    return out
