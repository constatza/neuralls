"""The tables stored beside group figures: the workload and full-cost builders and per-kind validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import assert_never

import polars as pl

from neuralls.domain.solver.models.cost import (
    GenerationCost,
    NotApplicable,
    OfflineStages,
    TrainShareOfGeneration,
    Unavailable,
    recorded_seconds,
)
from neuralls.domain.solver.models.workload import MethodWorkloadSummary
from neuralls.domain.solver.reporting.full_cost import (
    Complete,
    OfflineUnavailable,
    full_cost_seconds,
)
from neuralls.platform.reporting.figure_metadata import (
    FULL_COST_BREAKDOWN_SCHEMA,
    TABLE_SCHEMAS,
    WORKLOAD_SCHEMA,
    TableKind,
)
from neuralls.platform.reporting.offline_cost_codec import fit_origin_kind, stage_status
from neuralls.shared.types import PreconditionerName


class InvalidFigureTableError(ValueError):
    """A table does not have the columns and dtypes of the kind it is stored or read as."""


def build_group_table(summaries: Iterable[MethodWorkloadSummary]) -> pl.DataFrame:
    """The workload table: one row per method drawn, ordered by method id."""
    rows = sorted(
        (
            {
                "method_id": str(summary.method),
                "n_planned_rhs": summary.n_planned_rhs,
                "n_attempted_rhs": summary.n_attempted_rhs,
                "n_converged": summary.n_converged,
                "workload_setup_s": summary.workload_setup_s,
                "workload_setup_amortized_s": summary.workload_setup_amortized_s,
                "workload_converged_solve_s": summary.workload_converged_solve_s,
                "workload_failed_solve_s": summary.workload_failed_solve_s,
                "workload_solve_s": summary.workload_solve_s,
                "workload_online_s": summary.workload_online_s,
                "workload_online_amortized_s": summary.workload_online_amortized_s,
            }
            for summary in summaries
        ),
        key=lambda row: row["method_id"],
    )
    return pl.DataFrame(rows, schema=WORKLOAD_SCHEMA)


class FullStatus(StrEnum):
    """Whether a method's full cost is known: every offline stage is, or does not apply."""

    COMPLETE = "complete"
    OFFLINE_UNAVAILABLE = "offline_unavailable"


def build_full_cost_breakdown_table(
    summaries: Iterable[MethodWorkloadSummary],
    offline: Mapping[PreconditionerName, OfflineStages],
) -> pl.DataFrame:
    """The full-cost breakdown table: one row per method drawn, ordered by method id.

    ``offline`` holds the offline stages of every method of ``summaries``; the online columns are
    the pair workload sums of the summary, and the full cost is `full_cost_seconds` of the two.
    """
    ordered = sorted(summaries, key=lambda summary: str(summary.method))
    rows = [_full_cost_breakdown_row(summary, offline[summary.method]) for summary in ordered]
    return pl.DataFrame(rows, schema=FULL_COST_BREAKDOWN_SCHEMA)


def _full_cost_breakdown_row(
    summary: MethodWorkloadSummary, offline: OfflineStages
) -> dict[str, str | float | None]:
    match full_cost_seconds(offline, summary.workload_online_s):
        case Complete(seconds=seconds):
            full_status = FullStatus.COMPLETE
            full_s: float | None = seconds
        case OfflineUnavailable():
            full_status = FullStatus.OFFLINE_UNAVAILABLE
            full_s = None
        case unreachable:
            assert_never(unreachable)
    return {
        "method_id": str(summary.method),
        "generation_s": recorded_seconds(offline.charged_generation()),
        "generation_status": stage_status(offline.charged_generation()).value,
        **_generation_charge(offline.generation),
        "fit_s": recorded_seconds(offline.fit),
        "fit_status": stage_status(offline.fit).value,
        "fit_origin": fit_origin_kind(offline.fit_origin()).value,
        "workload_setup_s": summary.workload_setup_s,
        "workload_solve_s": summary.workload_solve_s,
        "full_s": full_s,
        "full_status": full_status.value,
    }


def _generation_charge(generation: GenerationCost) -> dict[str, float | str | None]:
    """The whole dataset's generation seconds, the share of them charged and its folds, when known."""
    match generation:
        case TrainShareOfGeneration(dataset=dataset, share=share):
            return {
                "generation_dataset_s": dataset.wall_time_seconds,
                "generation_share": share.value,
                "generation_folds": share.folds.value,
            }
        case NotApplicable() | Unavailable():
            return {
                "generation_dataset_s": None,
                "generation_share": None,
                "generation_folds": None,
            }
        case unreachable:
            assert_never(unreachable)


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
