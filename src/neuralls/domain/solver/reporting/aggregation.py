"""Matrix-first event aggregation and failure-aware report inputs."""

from __future__ import annotations

import statistics
from collections.abc import Collection, Mapping, Sequence
from math import fsum

import polars as pl

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
from neuralls.domain.solver.models.statistics import IterationStats
from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
    MethodWorkloadSummary,
)
from neuralls.domain.solver.reporting.event_frames import build_setup_frame, build_solve_frame
from neuralls.domain.solver.reporting.report_levels import (
    ReportLevel,
    label_for_tags,
    report_key,
)
from neuralls.shared.types import MatrixId, PreconditionerName, SetupEventId, SystemId

_PLANNED_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "dataset": pl.String,
    "matrix_id": pl.Int64,
    "system_id": pl.String,
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
        .with_columns(allocated_setup_s=pl.col("setup_s") / pl.col("planned_rhs_count"))
        .with_columns(amortized_online_s=pl.col("solve_s") + pl.col("allocated_setup_s"))
    )


def build_matrix_bundle_frame(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    *,
    allocation_planned: Sequence[SystemTag] | None = None,
) -> pl.DataFrame:
    """Build one setup-aware online-cost row per method and matrix.

    ``planned`` selects the report membership; ``allocation_planned`` keeps a
    matrix-set-and-RHS-type group's setup share tied to the complete matrix RHS bundle.
    """
    _validate_event_links(planned, setup_events, attempts)
    planned_counts = _planned_matrix_counts(planned)
    allocation_counts = _planned_matrix_counts(allocation_planned or planned).rename(
        {"planned_rhs_count": "allocation_rhs_count"}
    )
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
        .join(allocation_counts, on=["dataset", "matrix_id"], how="left", validate="m:1")
        .join(planned_counts, on=["dataset", "matrix_id"], how="left", validate="m:1")
        .join(solve_totals, on="setup_event_id", how="left", validate="1:1")
        .with_columns(
            pl.col("attempted_rhs_count").fill_null(0),
            pl.col("converged_rhs_count").fill_null(0),
            pl.col("solve_total_s").fill_null(0.0),
        )
        .with_columns(
            allocated_setup_s=pl.col("setup_s")
            * pl.col("planned_rhs_count")
            / pl.col("allocation_rhs_count"),
            matrix_online_s=pl.col("setup_s") + pl.col("solve_total_s"),
        )
    )


def summarize_method_workload(
    method: PreconditionerName,
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    workload_peak_memory: PeakMemory,
    *,
    allocation_planned: Sequence[SystemTag] | None = None,
) -> MethodWorkloadSummary:
    """Derive one complete or explicitly incomplete summary for ``method``."""
    method_setups = tuple(event for event in setup_events if event.method == method)
    method_attempts = tuple(attempt for attempt in attempts if attempt.method == method)
    bundles = build_matrix_bundle_frame(
        planned,
        method_setups,
        method_attempts,
        allocation_planned=allocation_planned,
    )
    attempted_ids = {attempt.system_id for attempt in method_attempts}
    missing_ids = tuple(tag.system_id for tag in planned if tag.system_id not in attempted_ids)
    setup_total_s = fsum(bundles["allocated_setup_s"])
    solve_total_s = fsum(bundles["solve_total_s"])
    workload_online_s = fsum(
        setup_s + solve_s
        for setup_s, solve_s in zip(
            bundles["allocated_setup_s"], bundles["solve_total_s"], strict=True
        )
    )
    n_attempted_rhs = len(method_attempts)
    n_converged = sum(isinstance(attempt.outcome, SolveConverged) for attempt in method_attempts)
    online_s_per_attempted_rhs = (
        workload_online_s / n_attempted_rhs if n_attempted_rhs > 0 else None
    )
    summary_fields = {
        "method": method,
        "n_planned_rhs": len(planned),
        "n_attempted_rhs": n_attempted_rhs,
        "n_converged": n_converged,
        "setup_total_s": setup_total_s,
        "solve_total_s": solve_total_s,
        "workload_online_s": workload_online_s,
        "online_s_per_attempted_rhs": online_s_per_attempted_rhs,
        "workload_peak_memory": workload_peak_memory,
    }
    if missing_ids:
        return IncompleteWorkloadSummary(**summary_fields, missing_system_ids=missing_ids)
    return CompleteWorkloadSummary(**summary_fields)


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


def convergence_by_level(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    level: ReportLevel,
    *,
    planned_by_method: Mapping[PreconditionerName, Sequence[SystemTag]] | None = None,
) -> dict[GroupLabel, dict[PreconditionerName, ConvergenceSummary]]:
    """Count planned, attempted, and converged RHS values for every planned method."""
    _validate_event_links(planned, setup_events, attempts)
    attempts_by_system = _attempts_by_system(attempts)
    report: dict[GroupLabel, dict[PreconditionerName, ConvergenceSummary]] = {}
    for tags in _tags_by_report_key(planned, level).values():
        label = label_for_tags(level, tags)
        systems = {tag.system_id for tag in tags}
        methods = _group_methods(tags, setup_events, planned_by_method)
        by_method: dict[PreconditionerName, ConvergenceSummary] = {}
        for method in methods:
            method_attempts = [
                attempt
                for system_id in systems
                for attempt in attempts_by_system.get(system_id, ())
                if attempt.method == method
            ]
            converged_iterations = [
                attempt.outcome.iterations
                for attempt in method_attempts
                if isinstance(attempt.outcome, SolveConverged)
            ]
            by_method[method] = ConvergenceSummary(
                n_planned_rhs=len(tags),
                n_attempted_rhs=len(method_attempts),
                n_converged=len(converged_iterations),
                iterations=_iteration_stats(converged_iterations),
            )
        if by_method:
            report[label] = by_method
    return report


def outcomes_by_system(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    level: ReportLevel,
    *,
    planned_by_method: Mapping[PreconditionerName, Sequence[SystemTag]] | None = None,
) -> dict[GroupLabel, dict[PreconditionerName, dict[SystemId, SystemOutcome]]]:
    """Build one censored outcome for every planned method/system pair."""
    _validate_event_links(planned, setup_events, attempts)
    planned_counts = {
        (row["dataset"], MatrixId(row["matrix_id"])): int(row["planned_rhs_count"])
        for row in _planned_matrix_counts(planned).iter_rows(named=True)
    }
    setups = {(event.dataset, event.matrix_id, event.method): event for event in setup_events}
    attempts_by_key = {(attempt.system_id, attempt.method): attempt for attempt in attempts}
    groups = _tags_by_report_key(planned, level)
    labels = {key: label_for_tags(level, tags) for key, tags in groups.items()}
    report: dict[GroupLabel, dict[PreconditionerName, dict[SystemId, SystemOutcome]]] = {}
    for key, tags in groups.items():
        label = labels[key]
        methods = _group_methods(tags, setup_events, planned_by_method)
        for method in methods:
            method_outcomes = report.setdefault(label, {}).setdefault(method, {})
            for tag in tags:
                attempt = attempts_by_key.get((tag.system_id, method))
                setup = setups.get((tag.dataset, tag.matrix_id, method))
                if attempt is None:
                    reason = (
                        setup.outcome.reason
                        if setup is not None and isinstance(setup.outcome, SetupFailed)
                        else "setup not recorded"
                        if setup is None
                        else "not attempted"
                    )
                    method_outcomes[tag.system_id] = Failed(
                        matrix_id=tag.matrix_id, iterations=0, reason=reason
                    )
                    continue
                if isinstance(attempt.outcome, SolveConverged):
                    if setup is None:
                        raise ValueError("solved attempt is missing its setup event")
                    method_outcomes[tag.system_id] = Solved(
                        matrix_id=attempt.matrix_id,
                        iterations=attempt.outcome.iterations,
                        solve_s=attempt.solve_s,
                        amortized_online_s=(
                            attempt.solve_s
                            + setup.setup_s / planned_counts[(tag.dataset, tag.matrix_id)]
                        ),
                        dataset=tag.dataset,
                    )
                else:
                    method_outcomes[tag.system_id] = Failed(
                        matrix_id=attempt.matrix_id,
                        iterations=attempt.outcome.iterations,
                        reason=attempt.outcome.reason,
                    )
    return report


def _group_methods(
    tags: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    planned_by_method: Mapping[PreconditionerName, Sequence[SystemTag]] | None,
) -> tuple[PreconditionerName, ...]:
    """Retain methods planned for this group even when execution never reached setup."""
    if planned_by_method is not None:
        group_ids = {tag.system_id for tag in tags}
        return tuple(
            method
            for method, method_planned in planned_by_method.items()
            if any(tag.system_id in group_ids for tag in method_planned)
        )
    matrices = {(tag.dataset, tag.matrix_id) for tag in tags}
    return tuple(
        dict.fromkeys(
            event.method for event in setup_events if (event.dataset, event.matrix_id) in matrices
        )
    )


def _iteration_stats(converged_iterations: Sequence[int]) -> IterationStats | None:
    if not converged_iterations:
        return None
    std = statistics.stdev(converged_iterations) if len(converged_iterations) > 1 else None
    return IterationStats(
        mean=statistics.fmean(converged_iterations),
        median=statistics.median(converged_iterations),
        std=std,
    )


def _tags_by_report_key(
    planned: Sequence[SystemTag], level: ReportLevel
) -> dict[tuple[str, ...], list[SystemTag]]:
    grouped: dict[tuple[str, ...], list[SystemTag]] = {}
    for tag in planned:
        grouped.setdefault(report_key(level, tag), []).append(tag)
    return grouped


def _attempts_by_system(
    attempts: Collection[SolveAttempt],
) -> dict[SystemId, list[SolveAttempt]]:
    grouped: dict[SystemId, list[SolveAttempt]] = {}
    for attempt in attempts:
        grouped.setdefault(attempt.system_id, []).append(attempt)
    return grouped


def _planned_matrix_counts(planned: Sequence[SystemTag]) -> pl.DataFrame:
    rows = [
        {"dataset": tag.dataset, "matrix_id": tag.matrix_id, "system_id": tag.system_id}
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
        if (attempt.dataset, attempt.recipe, attempt.matrix_id, attempt.rhs_ordinal) != (
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
