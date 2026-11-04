"""A figure's table is read back only against the kind of table its recipe writes.

The tests marked ``xfail`` pin the table interface that does not exist yet: one schema per
``figure_metadata.TableKind`` in ``figure_metadata.TABLE_SCHEMAS`` and
``figure_table.read_group_table(path, kind)``. They are
reached through module attributes so the file imports before those exist; strict ``xfail`` fails
loudly once one starts to pass, which is the cue to drop its marker.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    SPEED_RATIO_SCHEMA,
    speed_ratio_summary,
)
from neuralls.domain.benchmarking.profiles import PROFILE_SERIES_SCHEMA, ecdf_series
from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import SUMMARY_SCHEMA, Population, summarize
from neuralls.platform.reporting import figure_metadata, figure_planning, figure_table
from neuralls.platform.reporting.figure_planning import (
    FIGURE_RECIPES,
    GroupFigureRecipe,
    group_recipe,
)
from neuralls.platform.reporting.figure_table import (
    InvalidFigureTableError,
    build_group_table,
    write_group_table,
)
from neuralls.shared.types import PreconditionerName

TABLE_KINDS = ("SUMMARY", "SERIES", "SPEED_RATIO", "WORKLOAD_TOTALS")
"""Member names of ``TableKind``; names, not members, so collecting the tests needs no enum."""

_NO_TABLE_KINDS = pytest.mark.xfail(
    strict=True,
    raises=(AttributeError, TypeError),
    reason="tables are not validated per kind yet (figure_metadata.TABLE_SCHEMAS and "
    "read_group_table(path, kind) are missing)",
)
_NO_SUMMARY_PLANNING = pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="planning does not compute a summary frame per figure variant yet "
    "(figure_planning.figure_summaries is missing)",
)


@pytest.fixture
def table_frames(profile_report: RecipeProfileReport) -> dict[str, pl.DataFrame]:
    """One conforming frame of each table kind, from the domain statistics of one report."""
    observations = profile_report.observations
    return {
        "SUMMARY": summarize(observations, MetricId.ITERATIONS, Population.OWN),
        "SERIES": ecdf_series(observations, MetricId.ONLINE_S).frame,
        "SPEED_RATIO": speed_ratio_summary(
            observations, PreconditionerName("jacobi"), REFERENCE_METRIC
        ),
        "WORKLOAD_TOTALS": build_group_table(profile_report.summaries),
    }


@_NO_TABLE_KINDS
def test_there_is_exactly_one_schema_per_table_kind() -> None:
    assert set(figure_metadata.TABLE_SCHEMAS) == set(figure_metadata.TableKind)
    assert {kind.name for kind in figure_metadata.TableKind} == set(TABLE_KINDS)


@_NO_TABLE_KINDS
def test_each_table_kind_has_the_schema_of_the_frame_it_stores() -> None:
    schemas = figure_metadata.TABLE_SCHEMAS
    kind = figure_metadata.TableKind

    assert schemas[kind.SUMMARY] == SUMMARY_SCHEMA
    assert schemas[kind.SERIES] == PROFILE_SERIES_SCHEMA
    assert schemas[kind.SPEED_RATIO] == SPEED_RATIO_SCHEMA
    assert schemas[kind.WORKLOAD_TOTALS] == dict(build_group_table(()).schema)


@_NO_TABLE_KINDS
@pytest.mark.parametrize("name", TABLE_KINDS)
def test_a_table_is_read_back_as_the_kind_it_was_written_for(
    tmp_path: Path, table_frames: dict[str, pl.DataFrame], name: str
) -> None:
    path = tmp_path / "table.parquet"
    write_group_table(path, table_frames[name])

    read = figure_table.read_group_table(path, figure_metadata.TableKind[name])

    assert_frame_equal(read, table_frames[name])


@_NO_TABLE_KINDS
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
        figure_table.read_group_table(path, figure_metadata.TableKind[expected])


@_NO_TABLE_KINDS
@pytest.mark.parametrize("name", TABLE_KINDS)
def test_a_table_with_the_right_columns_but_other_types_is_rejected(
    tmp_path: Path, table_frames: dict[str, pl.DataFrame], name: str
) -> None:
    path = tmp_path / "table.parquet"
    write_group_table(path, table_frames[name].with_columns(pl.col(pl.Int64).cast(pl.Float64)))

    with pytest.raises(InvalidFigureTableError):
        figure_table.read_group_table(path, figure_metadata.TableKind[name])


@_NO_SUMMARY_PLANNING
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
    kind: figure_metadata.FigureKind,
    stem: str,
) -> None:
    summary = figure_planning.figure_summaries(profile_report)[stem]
    path = tmp_path / f"{stem}.parquet"
    write_group_table(path, summary)
    own = group_recipe(kind).table

    assert_frame_equal(figure_table.read_group_table(path, own), summary)
    for other in figure_metadata.TableKind:
        if other is not own:
            with pytest.raises(InvalidFigureTableError):
                figure_table.read_group_table(path, other)
