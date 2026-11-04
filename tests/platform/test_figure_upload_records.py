"""A tabled figure is uploaded with a card that names the table stored beside it."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
from tests.fakes.artifact_sink import InMemoryArtifactSink

from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting.figure_card import card_name, decode_figure_card
from neuralls.platform.reporting.figure_metadata import (
    FigureKind,
    SiblingTable,
    StagedFigure,
    StagedTabledFigure,
)
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.tracking.figure_uploader import FigureUploader
from neuralls.shared.scientific_description.policy import NotationPolicy

GROUP_DIRECTORY = "group_report"


def test_the_table_beside_a_figure_is_the_one_its_card_names(
    tmp_path: Path,
    profile_report: RecipeProfileReport,
    repeated_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
) -> None:
    plan = next(
        plan
        for plan in figure_plans(profile_report, repeated_timing_presentation, "key", {})
        if plan.metadata.kind is FigureKind.ECDF_TIME and plan.metadata.metric is MetricId.ONLINE_S
    )
    table = pl.DataFrame({"method": ["amg", "jacobi"], "n_solved": [4, 9]})
    figure_path = tmp_path / plan.filename
    figure_path.write_bytes(b"")
    sink = InMemoryArtifactSink()
    uploader = FigureUploader("session-run", notation, sink)

    uploader.upload(
        GROUP_DIRECTORY, GROUP_DIRECTORY, StagedTabledFigure(StagedFigure(plan, figure_path), table)
    )
    uploader.write_index()

    card = decode_figure_card(
        tomllib.loads(sink.files[GROUP_DIRECTORY, card_name(plan.filename)].decode("utf-8"))
    )
    match card.table:
        case SiblingTable(name=name):
            stored = pl.read_parquet(io.BytesIO(sink.files[GROUP_DIRECTORY, name]))
        case unreachable:
            raise AssertionError(
                f"a group figure card must name a sibling table, not {unreachable}"
            )
    assert_frame_equal(stored, table)
    assert sorted(file for _, file in sink.files) == sorted(
        [plan.filename, card_name(plan.filename), name, "figure_index.toml"]
    )
