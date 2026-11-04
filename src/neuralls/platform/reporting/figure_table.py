"""The tables stored beside group figures: the workload totals builder and per-kind validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from neuralls.domain.solver.models.workload import MethodWorkloadSummary
from neuralls.platform.reporting.figure_metadata import (
    TABLE_SCHEMAS,
    WORKLOAD_TOTALS_SCHEMA,
    TableKind,
)


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
                "setup_total_s": summary.setup_total_s,
                "solve_total_s": summary.solve_total_s,
                "workload_online_s": summary.workload_online_s,
            }
            for summary in summaries
        ),
        key=lambda row: row["method_id"],
    )
    return pl.DataFrame(rows, schema=WORKLOAD_TOTALS_SCHEMA)


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
