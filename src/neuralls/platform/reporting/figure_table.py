"""The tables stored beside group figures: the workload totals builder and per-kind validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import assert_never

import polars as pl

from neuralls.domain.solver.models.cost import OfflineStages, recorded_seconds
from neuralls.domain.solver.models.workload import MethodWorkloadSummary
from neuralls.domain.solver.reporting.total_cost import (
    Complete,
    OfflineUnavailable,
    total_cost_seconds,
)
from neuralls.platform.reporting.figure_metadata import (
    COST_BREAKDOWN_SCHEMA,
    TABLE_SCHEMAS,
    WORKLOAD_TOTALS_SCHEMA,
    TableKind,
)
from neuralls.platform.reporting.offline_cost_codec import stage_status
from neuralls.shared.types import PreconditionerName


class InvalidFigureTableError(ValueError):
    """A table does not have the columns and dtypes of the kind it is stored or read as."""


def build_group_table(summaries: Iterable[MethodWorkloadSummary]) -> pl.DataFrame:
    """The workload-totals table: one row per method drawn, ordered by method id."""
    rows = sorted(
        (
            {
                "method_id": str(summary.method),
                "n_planned_rhs": summary.n_planned_rhs,
                "n_attempted_rhs": summary.n_attempted_rhs,
                "n_converged": summary.n_converged,
                "allocated_setup_s": summary.allocated_setup_s,
                "converged_solve_s": summary.converged_solve_s,
                "failed_solve_s": summary.failed_solve_s,
                "solve_total_s": summary.solve_total_s,
                "workload_online_s": summary.workload_online_s,
            }
            for summary in summaries
        ),
        key=lambda row: row["method_id"],
    )
    return pl.DataFrame(rows, schema=WORKLOAD_TOTALS_SCHEMA)


class TotalStatus(StrEnum):
    """Whether a method's total cost is known: every offline stage is, or does not apply."""

    COMPLETE = "complete"
    OFFLINE_UNAVAILABLE = "offline_unavailable"


def build_cost_breakdown_table(
    summaries: Iterable[MethodWorkloadSummary],
    offline: Mapping[PreconditionerName, OfflineStages],
) -> pl.DataFrame:
    """The cost-breakdown table: one row per method drawn, ordered by method id.

    ``offline`` holds the offline stages of every method of ``summaries``; the online columns are
    the workload totals of the summary, and the total is `total_cost_seconds` of the two.
    """
    ordered = sorted(summaries, key=lambda summary: str(summary.method))
    rows = [_cost_breakdown_row(summary, offline[summary.method]) for summary in ordered]
    return pl.DataFrame(rows, schema=COST_BREAKDOWN_SCHEMA)


def _cost_breakdown_row(
    summary: MethodWorkloadSummary, offline: OfflineStages
) -> dict[str, str | float | None]:
    match total_cost_seconds(offline, summary.workload_online_s):
        case Complete(seconds=total_s):
            total_status = TotalStatus.COMPLETE
            total: float | None = total_s
        case OfflineUnavailable():
            total_status = TotalStatus.OFFLINE_UNAVAILABLE
            total = None
        case unreachable:
            assert_never(unreachable)
    return {
        "method_id": str(summary.method),
        "generation_s": recorded_seconds(offline.generation),
        "generation_status": stage_status(offline.generation).value,
        "training_s": recorded_seconds(offline.training),
        "training_status": stage_status(offline.training).value,
        "allocated_setup_s": summary.allocated_setup_s,
        "solve_total_s": summary.solve_total_s,
        "total_s": total,
        "total_status": total_status.value,
    }


def write_group_table(path: Path, table: pl.DataFrame) -> None:
    """Write ``table`` as parquet."""
    table.write_parquet(path)


def check_table_kind(table: pl.DataFrame, kind: TableKind) -> pl.DataFrame:
    """``table`` itself, once it has exactly the columns and dtypes of ``kind`` in order."""
    if table.schema != pl.Schema(TABLE_SCHEMAS[kind]):
        raise InvalidFigureTableError(f"the table does not have the {kind.value} table schema")
    return table


def read_group_table(path: Path, kind: TableKind) -> pl.DataFrame:
    """Read the table at ``path``, requiring exactly the schema of ``kind``."""
    return check_table_kind(pl.read_parquet(path), kind)
