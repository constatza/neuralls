"""Aggregation of the metrics frame into projections and per-system times per group."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from neuralls.domain.solver.models.labels import GroupLabel
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
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


def report_level(
    frame: pl.DataFrame,
    level: ReportLevel,
    sharing_by_name: Mapping[str, SetupSharingPolicy],
) -> dict[GroupLabel, dict[str, ProjectedComparisonResult]]:
    """Reduce the frame to one projection per (group, preconditioner).

    Args:
        frame: Output of `build_metrics_frame`.
        level: Which columns, besides the preconditioner, key the groups.
        sharing_by_name: Each preconditioner's setup/generation sharing policy.

    Returns:
        Projections per group label, groups and preconditioners in first-seen order.

    Raises:
        KeyError: If a preconditioner in the frame has no sharing policy.
    """
    keys = level_columns(level)
    aggregated = frame.group_by([*keys, PRECONDITIONER], maintain_order=True).agg(
        n_systems=pl.len(),
        setup_total=pl.col("setup_s").sum(),
        generation_mean=pl.col("generation_s").mean(),
        solve_mean=pl.col("solve_s").mean(),
        solve_std=pl.col("solve_s").std(),
        iterations_total=pl.col("iterations").sum(),
        iterations_std=pl.col("iterations").std(),
    )
    mixtures = mixtures_by_key(frame, keys)
    report: dict[GroupLabel, dict[str, ProjectedComparisonResult]] = {}
    for row in aggregated.iter_rows(named=True):
        label = group_label(level, row, mixtures)
        n = row["n_systems"]
        generation_mean = row["generation_mean"]
        solve_mean = row["solve_mean"]
        name = row[PRECONDITIONER]
        report.setdefault(label, {})[name] = ProjectedComparisonResult(
            preconditioner_name=name,
            n_systems=n,
            setup_sharing=sharing_by_name[name],
            setup_cost_total_seconds=row["setup_total"],
            # Mean over the available results scaled to n, so the per-system
            # average (total / n) is that mean rather than diluted by
            # unavailable results.
            generation_cost_total_seconds=(
                None if generation_mean is None else generation_mean * n
            ),
            total_solve_time_seconds=0.0 if solve_mean is None else solve_mean * n,
            total_iterations=row["iterations_total"],
            solve_time_std_seconds=row["solve_std"],
            iterations_std=row["iterations_std"],
        )
    return report


def time_to_solution_by_system(
    frame: pl.DataFrame, level: ReportLevel
) -> dict[GroupLabel, dict[str, dict[str, float]]]:
    """Each group's per-preconditioner, per-system time-to-solution.

    Systems whose time is incomparable (null) are omitted, in frame order.
    """
    keys = level_columns(level)
    mixtures = mixtures_by_key(frame, keys)
    comparable = frame.filter(pl.col("time_to_solution_s").is_not_null())
    out: dict[GroupLabel, dict[str, dict[str, float]]] = {}
    for row in comparable.iter_rows(named=True):
        label = group_label(level, row, mixtures)
        out.setdefault(label, {}).setdefault(row[PRECONDITIONER], {})[row[SYSTEM_ID]] = row[
            "time_to_solution_s"
        ]
    return out
