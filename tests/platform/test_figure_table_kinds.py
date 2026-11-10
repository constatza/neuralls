"""A figure's table is read back only against the kind of table its recipe writes."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.figure_support import no_offline_cost

from neuralls.domain.benchmarking.cost_curves import COST_CURVE_SCHEMA, break_even_curves
from neuralls.domain.benchmarking.paired import (
    PAIRED_SCHEMA,
    REFERENCE_METRIC,
    SPEED_RATIO_SCHEMA,
    paired_systems,
    speed_ratio_summary,
)
from neuralls.domain.benchmarking.profiles import PROFILE_SERIES_SCHEMA, ecdf_series
from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.reporting.cohorts import (
    COHORT_RATIO_OF_SUMS_SCHEMA,
    COHORT_SUMMARY_SCHEMA,
    CohortRows,
    by_cohort,
)
from neuralls.domain.solver.reporting.matrix_breakdown import (
    MATRIX_BREAKDOWN_SCHEMA,
    matrix_breakdown,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import (
    RATIO_OF_SUMS_SCHEMA,
    SUMMARY_SCHEMA,
    Population,
    summarize,
    time_per_iteration_summary,
)
from neuralls.platform.reporting.figure_metadata import TABLE_SCHEMAS, FigureKind, TableKind
from neuralls.platform.reporting.figure_recipes import (
    FIGURE_RECIPES,
    GroupFigureRecipe,
    group_recipe,
)
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.platform.reporting.figure_table import (
    InvalidFigureTableError,
    build_cost_breakdown_table,
    build_group_table,
    read_group_table,
    write_group_table,
)
from neuralls.shared.types import PreconditionerName

JACOBI = PreconditionerName("jacobi")

TABLE_KINDS = tuple(kind.name for kind in TableKind)


@pytest.fixture
def table_frames(profile_report: RecipeProfileReport) -> dict[str, pl.DataFrame]:
    """One conforming frame of each table kind, from the domain statistics of one report."""
    observations = profile_report.observations
    return {
        "SUMMARY": summarize(observations, MetricId.ITERATIONS, Population.OWN),
        "SERIES": ecdf_series(observations, MetricId.ONLINE_S).frame,
        "PAIRED": paired_systems(observations, JACOBI, REFERENCE_METRIC),
        "SPEED_RATIO": speed_ratio_summary(observations, JACOBI, REFERENCE_METRIC),
        "WORKLOAD_TOTALS": build_group_table(profile_report.summaries),
        "COST_CURVES": break_even_curves(observations, JACOBI),
        "COHORT_SUMMARY": by_cohort(
            observations,
            lambda frame: summarize(frame, MetricId.SOLVE_S, Population.OWN),
            CohortRows.RECIPES_AND_MIXTURE,
            schema=SUMMARY_SCHEMA,
        ),
        "COHORT_RATIO_OF_SUMS": by_cohort(
            observations,
            lambda frame: time_per_iteration_summary(frame, Population.OWN),
            CohortRows.RECIPES,
            schema=RATIO_OF_SUMS_SCHEMA,
        ),
        "MATRIX_BREAKDOWN": matrix_breakdown(observations, MetricId.ITERATIONS),
        "COST_BREAKDOWN": build_cost_breakdown_table(
            profile_report.summaries, no_offline_cost(profile_report)
        ),
    }


def test_there_is_exactly_one_schema_per_table_kind() -> None:
    assert set(TABLE_SCHEMAS) == set(TableKind)
    assert {kind.name for kind in TableKind} == set(TABLE_KINDS)


def test_each_table_kind_has_the_schema_of_the_frame_it_stores() -> None:
    schemas = TABLE_SCHEMAS
    kind = TableKind

    assert schemas[kind.SUMMARY] == SUMMARY_SCHEMA
    assert schemas[kind.SERIES] == PROFILE_SERIES_SCHEMA
    assert schemas[kind.PAIRED] == PAIRED_SCHEMA
    assert schemas[kind.SPEED_RATIO] == SPEED_RATIO_SCHEMA
    assert schemas[kind.WORKLOAD_TOTALS] == dict(build_group_table(()).schema)
    assert schemas[kind.COST_CURVES] == COST_CURVE_SCHEMA
    assert schemas[kind.COHORT_SUMMARY] == COHORT_SUMMARY_SCHEMA
    assert schemas[kind.COHORT_RATIO_OF_SUMS] == COHORT_RATIO_OF_SUMS_SCHEMA
    assert schemas[kind.MATRIX_BREAKDOWN] == MATRIX_BREAKDOWN_SCHEMA
    assert schemas[kind.COST_BREAKDOWN] == dict(build_cost_breakdown_table((), {}).schema)


@pytest.mark.parametrize("name", TABLE_KINDS)
def test_a_table_is_read_back_as_the_kind_it_was_written_for(
    tmp_path: Path, table_frames: dict[str, pl.DataFrame], name: str
) -> None:
    path = tmp_path / "table.parquet"
    write_group_table(path, table_frames[name])

    read = read_group_table(path, TableKind[name])

    assert_frame_equal(read, table_frames[name])


@pytest.mark.parametrize(
    ("written", "expected"),
    [pytest.param(a, b, id=f"{a}-as-{b}") for a, b in permutations(TABLE_KINDS, 2)],
)
def test_a_table_is_rejected_when_read_as_another_kind(
    tmp_path: Path, table_frames: dict[str, pl.DataFrame], written: str, expected: str
) -> None:
    path = tmp_path / "table.parquet"
    write_group_table(path, table_frames[written])

    with pytest.raises(InvalidFigureTableError):
        read_group_table(path, TableKind[expected])


@pytest.mark.parametrize("name", TABLE_KINDS)
def test_a_table_with_the_right_columns_but_other_types_is_rejected(
    tmp_path: Path, table_frames: dict[str, pl.DataFrame], name: str
) -> None:
    path = tmp_path / "table.parquet"
    retyped = table_frames[name].with_columns(
        pl.col(pl.Int64).cast(pl.Float64), pl.col(pl.Float64).cast(pl.Float32)
    )
    write_group_table(path, retyped)

    with pytest.raises(InvalidFigureTableError):
        read_group_table(path, TableKind[name])


@pytest.mark.parametrize(
    ("kind", "stem"),
    [
        pytest.param(recipe.kind, variant.stem, id=variant.stem)
        for recipe in FIGURE_RECIPES.values()
        if isinstance(recipe, GroupFigureRecipe) and RecipeProfileReport in recipe.applies_to
        for variant in recipe.variants
    ],
)
def test_the_table_planned_for_a_recipe_is_read_back_only_as_that_recipes_kind(
    tmp_path: Path,
    profile_report: RecipeProfileReport,
    kind: FigureKind,
    stem: str,
) -> None:
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))[stem]
    path = tmp_path / f"{stem}.parquet"
    write_group_table(path, summary)
    own = group_recipe(kind).table

    assert_frame_equal(read_group_table(path, own), summary)
    for other in TableKind:
        if other is not own:
            with pytest.raises(InvalidFigureTableError):
                read_group_table(path, other)
