"""The workload-totals table and the table each group figure kind stores beside itself."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from neuralls.domain.solver.models.benchmark_events import UnknownPeakMemory
from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    FitFolds,
    Known,
    OfflineStages,
    TrainShare,
    TrainShareOfGeneration,
    Unavailable,
)
from neuralls.domain.solver.models.workload import CompleteWorkloadSummary, MethodWorkloadSummary
from neuralls.platform.reporting.figure_metadata import (
    FULL_COST_BREAKDOWN_SCHEMA,
    GROUP_FIGURE_KINDS,
    WORKLOAD_SCHEMA,
    FigureKind,
    NoTable,
    SiblingTable,
    TableKind,
)
from neuralls.platform.reporting.figure_planning import table_of
from neuralls.platform.reporting.figure_recipes import group_recipe
from neuralls.platform.reporting.figure_table import (
    InvalidFigureTableError,
    build_full_cost_breakdown_table,
    build_group_table,
    read_group_table,
    write_group_table,
)
from neuralls.shared.types import CostProvenance, PreconditionerName


def _summary(
    name: str, converged: int, setup: float, solve: float, failed: float
) -> MethodWorkloadSummary:
    """Four RHS on two matrices: each pair is charged twice the amortized setup share."""
    return CompleteWorkloadSummary(
        method=PreconditionerName(name),
        n_planned_rhs=4,
        n_attempted_rhs=4,
        n_converged=converged,
        workload_setup_s=2 * setup,
        workload_setup_amortized_s=setup,
        workload_solve_s=solve,
        workload_failed_solve_s=failed,
        workload_online_s=2 * setup + solve,
        workload_online_amortized_s=setup + solve,
        workload_peak_memory=UnknownPeakMemory(),
    )


@pytest.fixture
def summaries() -> tuple[MethodWorkloadSummary, ...]:
    """``pcg_jacobi`` spent 0.75 of its 2 solve seconds on its one failed attempt."""
    return (_summary("pcg_jacobi", 3, 0.5, 2.0, 0.75), _summary("amg", 4, 1.0, 1.5, 0.0))


@pytest.fixture
def only_failures() -> MethodWorkloadSummary:
    """A method whose four attempts all failed, after 3 seconds of solving."""
    return _summary("broken", 0, 1.0, 3.0, 3.0)


def test_table_has_one_row_per_method_ordered_by_id(
    summaries: tuple[MethodWorkloadSummary, ...],
) -> None:
    table = build_group_table(summaries)

    assert dict(table.schema) == WORKLOAD_SCHEMA
    assert table.to_dicts() == [
        {
            "method_id": "amg",
            "n_planned_rhs": 4,
            "n_attempted_rhs": 4,
            "n_converged": 4,
            "workload_setup_s": 2.0,
            "workload_setup_amortized_s": 1.0,
            "workload_converged_solve_s": 1.5,
            "workload_failed_solve_s": 0.0,
            "workload_solve_s": 1.5,
            "workload_online_s": 3.5,
            "workload_online_amortized_s": 2.5,
        },
        {
            "method_id": "pcg_jacobi",
            "n_planned_rhs": 4,
            "n_attempted_rhs": 4,
            "n_converged": 3,
            "workload_setup_s": 1.0,
            "workload_setup_amortized_s": 0.5,
            "workload_converged_solve_s": 1.25,
            "workload_failed_solve_s": 0.75,
            "workload_solve_s": 2.0,
            "workload_online_s": 3.0,
            "workload_online_amortized_s": 2.5,
        },
    ]


def test_the_converged_and_failed_solve_times_sum_to_the_totals(
    summaries: tuple[MethodWorkloadSummary, ...], only_failures: MethodWorkloadSummary
) -> None:
    table = build_group_table((*summaries, only_failures))

    solve = pl.col("workload_converged_solve_s") + pl.col("workload_failed_solve_s")
    split = table.select(
        "method_id",
        solve=solve,
        online=pl.col("workload_setup_s") + solve,
        online_amortized=pl.col("workload_setup_amortized_s") + solve,
    )
    totals = table.select(
        "method_id",
        solve="workload_solve_s",
        online="workload_online_s",
        online_amortized="workload_online_amortized_s",
    )
    assert_frame_equal(split, totals)


def test_a_method_with_only_failures_has_no_converged_solve_time(
    only_failures: MethodWorkloadSummary,
) -> None:
    table = build_group_table((only_failures,))

    assert table.select(
        "n_converged", "workload_converged_solve_s", "workload_failed_solve_s"
    ).rows() == [(0, 0.0, 3.0)]


def test_cost_breakdown_stacks_each_methods_offline_stages_on_its_online_totals(
    summaries: tuple[MethodWorkloadSummary, ...],
) -> None:
    offline = {
        PreconditionerName("pcg_jacobi"): NO_OFFLINE_COST,
        PreconditionerName("amg"): OfflineStages(
            generation=TrainShareOfGeneration(
                Known(20.0, None, CostProvenance.HISTORICAL), TrainShare(0.5, FitFolds.TRAIN)
            ),
            fit=Known(90.0, None, CostProvenance.HISTORICAL),
        ),
    }

    table = build_full_cost_breakdown_table(summaries, offline)

    assert dict(table.schema) == FULL_COST_BREAKDOWN_SCHEMA
    assert table.to_dicts() == [
        {
            "method_id": "amg",
            "generation_s": 10.0,
            "generation_status": "known",
            "dataset_generation_s": 20.0,
            "generation_share": 0.5,
            "generation_folds": "train",
            "fit_s": 90.0,
            "fit_status": "known",
            "fit_origin": "job",
            "workload_setup_s": 2.0,
            "workload_solve_s": 1.5,
            "full_s": 103.5,
            "full_status": "complete",
        },
        {
            "method_id": "pcg_jacobi",
            "generation_s": None,
            "generation_status": "not_applicable",
            "dataset_generation_s": None,
            "generation_share": None,
            "generation_folds": None,
            "fit_s": None,
            "fit_status": "not_applicable",
            "fit_origin": "none",
            "workload_setup_s": 1.0,
            "workload_solve_s": 2.0,
            "full_s": 3.0,
            "full_status": "complete",
        },
    ]


def test_cost_breakdown_states_a_fit_measured_in_the_comparison_as_inline(
    summaries: tuple[MethodWorkloadSummary, ...],
) -> None:
    offline = {
        PreconditionerName("pcg_jacobi"): NO_OFFLINE_COST,
        PreconditionerName("amg"): OfflineStages(
            generation=TrainShareOfGeneration(
                Known(10.0, None, CostProvenance.HISTORICAL), TrainShare(1.0, FitFolds.TRAIN)
            ),
            fit=Known(0.5, 64, CostProvenance.MEASURED),
        ),
    }

    (amg, _) = build_full_cost_breakdown_table(summaries, offline).to_dicts()

    assert (amg["fit_s"], amg["fit_origin"], amg["full_s"]) == (0.5, "inline", 14.0)


def test_cost_breakdown_has_no_full_cost_when_an_offline_stage_is_unavailable(
    summaries: tuple[MethodWorkloadSummary, ...],
) -> None:
    offline = {
        PreconditionerName("pcg_jacobi"): NO_OFFLINE_COST,
        PreconditionerName("amg"): OfflineStages(
            generation=TrainShareOfGeneration(
                Known(10.0, None, CostProvenance.HISTORICAL), TrainShare(1.0, FitFolds.TRAIN)
            ),
            fit=Unavailable("1 of 1 checkpoints have no recorded training run"),
        ),
    }

    (amg, _) = build_full_cost_breakdown_table(summaries, offline).to_dicts()

    assert amg == {
        "method_id": "amg",
        "generation_s": 10.0,
        "generation_status": "known",
        "dataset_generation_s": 10.0,
        "generation_share": 1.0,
        "generation_folds": "train",
        "fit_s": None,
        "fit_status": "unavailable",
        "fit_origin": "job",
        "workload_setup_s": 2.0,
        "workload_solve_s": 1.5,
        "full_s": None,
        "full_status": "offline_unavailable",
    }


def test_cost_breakdown_table_round_trips_through_parquet(
    tmp_path: Path, summaries: tuple[MethodWorkloadSummary, ...]
) -> None:
    offline = {
        PreconditionerName("pcg_jacobi"): OfflineStages(
            Unavailable("no manifest"), Known(3.0, None, CostProvenance.HISTORICAL)
        ),
        PreconditionerName("amg"): NO_OFFLINE_COST,
    }
    table = build_full_cost_breakdown_table(summaries, offline)
    path = tmp_path / "table.parquet"
    write_group_table(path, table)

    assert_frame_equal(read_group_table(path, TableKind.FULL_COST_BREAKDOWN), table)


def test_empty_cost_breakdown_has_the_schema() -> None:
    table = build_full_cost_breakdown_table((), {})

    assert table.height == 0
    assert dict(table.schema) == FULL_COST_BREAKDOWN_SCHEMA


def test_empty_summaries_give_an_empty_table_with_the_schema() -> None:
    table = build_group_table(())

    assert table.height == 0
    assert dict(table.schema) == WORKLOAD_SCHEMA


def test_table_round_trips_through_parquet(
    tmp_path: Path, summaries: tuple[MethodWorkloadSummary, ...]
) -> None:
    path = tmp_path / "table.parquet"
    table = build_group_table(summaries)

    write_group_table(path, table)

    assert read_group_table(path, TableKind.WORKLOAD).equals(table)


def test_a_table_with_another_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pl.DataFrame({"method_id": ["a"]}).write_parquet(path)

    with pytest.raises(InvalidFigureTableError):
        read_group_table(path, TableKind.WORKLOAD)


@pytest.mark.parametrize("kind", sorted(FigureKind))
def test_a_group_kind_has_a_same_stem_table_of_its_recipes_kind(kind: FigureKind) -> None:
    expected = (
        SiblingTable("fig.parquet", group_recipe(kind).table)
        if kind in GROUP_FIGURE_KINDS
        else NoTable()
    )

    assert table_of("fig", kind) == expected
