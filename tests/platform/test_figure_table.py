"""The plotted table of a group figure: built from summaries, round-tripped as parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from neuralls.domain.solver.models.benchmark_events import UnknownPeakMemory
from neuralls.domain.solver.models.workload import CompleteWorkloadSummary, MethodWorkloadSummary
from neuralls.platform.reporting.figure_metadata import (
    GROUP_FIGURE_KINDS,
    GROUP_TABLE_SCHEMA,
    FigureKind,
    NoTable,
    SiblingTable,
    table_of,
)
from neuralls.platform.reporting.figure_table import (
    InvalidFigureTableError,
    build_group_table,
    read_group_table,
    write_group_table,
)
from neuralls.shared.types import PreconditionerName


def _summary(name: str, converged: int, setup: float, solve: float) -> MethodWorkloadSummary:
    return CompleteWorkloadSummary(
        method=PreconditionerName(name),
        n_planned_rhs=4,
        n_attempted_rhs=4,
        n_converged=converged,
        setup_total_s=setup,
        solve_total_s=solve,
        workload_online_s=setup + solve,
        online_s_per_attempted_rhs=(setup + solve) / 4,
        workload_peak_memory=UnknownPeakMemory(),
    )


@pytest.fixture
def summaries() -> tuple[MethodWorkloadSummary, ...]:
    return (_summary("pcg_jacobi", 3, 0.5, 2.0), _summary("amg", 4, 1.0, 1.5))


def test_table_has_one_row_per_method_ordered_by_id(
    summaries: tuple[MethodWorkloadSummary, ...],
) -> None:
    table = build_group_table(summaries)

    assert dict(table.schema) == GROUP_TABLE_SCHEMA
    assert table.to_dicts() == [
        {
            "method_id": "amg",
            "n_planned_rhs": 4,
            "n_attempted_rhs": 4,
            "n_converged": 4,
            "setup_total_s": 1.0,
            "solve_total_s": 1.5,
            "workload_online_s": 2.5,
        },
        {
            "method_id": "pcg_jacobi",
            "n_planned_rhs": 4,
            "n_attempted_rhs": 4,
            "n_converged": 3,
            "setup_total_s": 0.5,
            "solve_total_s": 2.0,
            "workload_online_s": 2.5,
        },
    ]


def test_empty_summaries_give_an_empty_table_with_the_schema() -> None:
    table = build_group_table(())

    assert table.height == 0
    assert dict(table.schema) == GROUP_TABLE_SCHEMA


def test_table_round_trips_through_parquet(
    tmp_path: Path, summaries: tuple[MethodWorkloadSummary, ...]
) -> None:
    path = tmp_path / "table.parquet"
    table = build_group_table(summaries)

    write_group_table(path, table)

    assert read_group_table(path).equals(table)


def test_a_table_with_another_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pl.DataFrame({"method_id": ["a"]}).write_parquet(path)

    with pytest.raises(InvalidFigureTableError):
        read_group_table(path)


@pytest.mark.parametrize("kind", sorted(FigureKind))
def test_only_group_kinds_have_a_same_stem_table(kind: FigureKind) -> None:
    expected = SiblingTable("fig.parquet") if kind in GROUP_FIGURE_KINDS else NoTable()

    assert table_of("fig.png", kind) == expected
