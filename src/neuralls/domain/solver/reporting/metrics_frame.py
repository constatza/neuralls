"""Long-form metrics frame: one row per preconditioner and system."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.reporting.solve_time import (
    available_generation_seconds,
    recurring_seconds,
    time_to_solution,
)

PRECONDITIONER = "preconditioner"

SYSTEM_ID = "system_id"

METRICS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "dataset": pl.String,
    "recipe": pl.String,
    "matrix_id": pl.Int64,
    "rhs_ordinal": pl.Int64,
    SYSTEM_ID: pl.String,
    PRECONDITIONER: pl.String,
    "setup_s": pl.Float64,
    "generation_s": pl.Float64,
    "solve_s": pl.Float64,
    "iterations": pl.Int64,
    "converged": pl.Boolean,
    "time_to_solution_s": pl.Float64,
    "recurring_s": pl.Float64,
    "peak_memory_bytes": pl.Int64,
}


def build_metrics_frame(
    results_by_system: Mapping[SystemTag, Mapping[str, CGComparisonResult]],
    sharing_by_name: Mapping[str, SetupSharingPolicy],
) -> pl.DataFrame:
    """One row per (system, preconditioner).

    ``time_to_solution_s`` is the full wall time of the row's result (its own
    copy of any shared one-time cost included); ``recurring_s`` is that time
    without the setup and generation the preconditioner's sharing policy flags
    as one-time, i.e. what the system itself costs. A setup that depends on
    each system's matrix stays in ``recurring_s``.

    ``converged`` is kept per row so every aggregate can say which runs it
    covers: a run stopped by the iteration cap is a censored observation, not
    a cheap solve.

    ``generation_s`` is null when the generation cost is unavailable (the same
    results ``time_to_solution_s`` treats as incomparable, which is null too);
    ``solve_s`` is null when the solve was not measured.

    Raises:
        KeyError: If a preconditioner has no sharing policy.
    """
    rows = [
        {
            "dataset": tag.dataset,
            "recipe": tag.recipe,
            "matrix_id": tag.matrix_id,
            "rhs_ordinal": tag.rhs_ordinal,
            SYSTEM_ID: tag.system_id,
            PRECONDITIONER: name,
            "setup_s": result.setup_cost,
            "generation_s": available_generation_seconds(result),
            "solve_s": result.solve_time_seconds,
            "iterations": result.iterations,
            "converged": result.converged,
            "time_to_solution_s": time_to_solution(result),
            "recurring_s": recurring_seconds(result, sharing_by_name[name]),
            "peak_memory_bytes": result.peak_memory_bytes,
        }
        for tag, system_results in results_by_system.items()
        for name, result in system_results.items()
    ]
    return pl.DataFrame(rows, schema=METRICS_SCHEMA)
