"""A tabled figure is uploaded with a card that names the table stored beside it."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.fakes.artifact_sink import InMemoryArtifactSink
from tests.figure_support import no_offline_cost

from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.split_record import DatasetSplitRecord
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting.figure_card import card_name, decode_figure_card
from neuralls.platform.reporting.figure_formats import FigureTarget, figure_files
from neuralls.platform.reporting.figure_metadata import (
    FigureKind,
    SiblingTable,
    StagedFigure,
    StagedTabledFigure,
)
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.platform.reporting.figure_table import InvalidFigureTableError
from neuralls.platform.tracking.figure_uploader import FigureUploader
from neuralls.platform.tracking.split_records import SPLITS_ARTIFACT, encode_split_records
from neuralls.shared.scientific_description.policy import NotationPolicy

GROUP_DIRECTORY = "group_report"


def _written_target(directory: Path, stem: str) -> FigureTarget:
    """A target whose files exist, as the figure writer leaves them (empty: images are not tested)."""
    target = FigureTarget(directory, stem)
    for path in target.paths:
        path.write_bytes(b"")
    return target


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
    table = figure_summaries(profile_report, no_offline_cost(profile_report))["ecdf_time_online"]
    target = _written_target(tmp_path, plan.stem)
    sink = InMemoryArtifactSink()
    uploader = FigureUploader("session-run", notation, sink)

    uploader.upload(
        GROUP_DIRECTORY, GROUP_DIRECTORY, StagedTabledFigure(StagedFigure(plan, target), table)
    )
    uploader.write_index(())

    card = decode_figure_card(
        tomllib.loads(sink.files[GROUP_DIRECTORY, card_name(plan.stem)].decode("utf-8"))
    )
    match card.table:
        case SiblingTable(name=name):
            stored = pl.read_parquet(io.BytesIO(sink.files[GROUP_DIRECTORY, name]))
        case unreachable:
            raise AssertionError(
                f"a group figure card must name a sibling table, not {unreachable}"
            )
    assert_frame_equal(stored, table)
    assert card.files == figure_files(plan.stem)
    assert sorted(file for _, file in sink.files) == sorted(
        [*card.files, card_name(plan.stem), name, "figure_index.toml", "splits.toml"]
    )


def test_a_table_of_another_kind_than_the_cards_is_not_uploaded(
    tmp_path: Path,
    profile_report: RecipeProfileReport,
    repeated_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
) -> None:
    plan = next(
        plan
        for plan in figure_plans(profile_report, repeated_timing_presentation, "key", {})
        if plan.metadata.kind is FigureKind.ECDF_ITERATIONS
    )
    target = _written_target(tmp_path, plan.stem)
    sink = InMemoryArtifactSink()
    uploader = FigureUploader("session-run", notation, sink)
    other_kind = figure_summaries(profile_report, no_offline_cost(profile_report))[
        "solve_time_bars"
    ]

    with pytest.raises(InvalidFigureTableError):
        uploader.upload(
            GROUP_DIRECTORY,
            GROUP_DIRECTORY,
            StagedTabledFigure(StagedFigure(plan, target), other_kind),
        )


def test_the_index_cites_the_split_records_copied_to_the_report_root(
    notation: NotationPolicy, split_records: tuple[DatasetSplitRecord, ...]
) -> None:
    sink = InMemoryArtifactSink()
    uploader = FigureUploader("session-run", notation, sink)

    uploader.write_index(split_records)

    assert sink.files[None, SPLITS_ARTIFACT] == encode_split_records(split_records)
    assert tomllib.loads(sink.files[None, "figure_index.toml"].decode("utf-8"))["splits"] == (
        SPLITS_ARTIFACT
    )
