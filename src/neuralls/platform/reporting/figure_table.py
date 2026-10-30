"""The compact plotted table of a group figure: one row per method with its workload totals."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from neuralls.domain.solver.models.workload import MethodWorkloadSummary
from neuralls.platform.reporting.figure_metadata import GROUP_TABLE_SCHEMA


class InvalidFigureTableError(ValueError):
    """A stored plotted table does not have the group table schema."""


def build_group_table(summaries: Iterable[MethodWorkloadSummary]) -> pl.DataFrame:
    """One row per method drawn, ordered by method id; no histories or per-system series."""
    rows = sorted(
        (
            {
                "method_id": str(summary.method),
                "n_planned_rhs": summary.n_planned_rhs,
                "n_attempted_rhs": summary.n_attempted_rhs,
                "n_converged": summary.n_converged,
                "setup_total_s": summary.setup_total_s,
                "solve_total_s": summary.solve_total_s,
                "workload_online_s": summary.workload_online_s,
            }
            for summary in summaries
        ),
        key=lambda row: row["method_id"],
    )
    return pl.DataFrame(rows, schema=GROUP_TABLE_SCHEMA)


def write_group_table(path: Path, table: pl.DataFrame) -> None:
    """Write ``table`` as parquet."""
    table.write_parquet(path)


def read_group_table(path: Path) -> pl.DataFrame:
    """Read the table at ``path``, requiring exactly the group table schema."""
    table = pl.read_parquet(path)
    if dict(table.schema) != GROUP_TABLE_SCHEMA:
        raise InvalidFigureTableError("the table does not have the group table schema")
    return table
