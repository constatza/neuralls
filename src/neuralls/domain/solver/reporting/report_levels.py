"""The fine, medium and global report levels, and the labels and keys they induce."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.labels import FineLabel, GlobalLabel, GroupLabel, MediumLabel
from neuralls.domain.solver.reporting.metrics_frame import (
    SYSTEM_ID,
)
from neuralls.shared.types import DatasetId


@dataclass(frozen=True, slots=True)
class FineLevel:
    """Group by dataset and recipe: one homogeneous problem family per group."""


@dataclass(frozen=True, slots=True)
class MediumLevel:
    """Group by dataset, pooling that dataset's recipes."""


@dataclass(frozen=True, slots=True)
class GlobalLevel:
    """No keys: pool every system in the case."""


type ReportLevel = FineLevel | MediumLevel | GlobalLevel


def report_key(level: ReportLevel, tag: SystemTag) -> tuple[str, ...]:
    """The values of ``tag`` that select its report group at ``level``."""
    match level:
        case FineLevel():
            return (tag.dataset, tag.recipe)
        case MediumLevel():
            return (tag.dataset,)
        case GlobalLevel():
            return ()


def label_key(label: GroupLabel) -> tuple[str, ...]:
    """The `report_key` that selects ``label``'s group (mixtures are not part of the key)."""
    match label:
        case FineLabel(dataset=dataset, recipe=recipe):
            return (dataset, recipe)
        case MediumLabel(dataset=dataset):
            return (dataset,)
        case GlobalLabel():
            return ()


def label_for_tags(level: ReportLevel, tags: Sequence[SystemTag]) -> GroupLabel:
    """The label of the group ``tags`` make up (all must share one `report_key`).

    Used for a group nothing was measured in, which `report_level` never sees.
    """
    mixture = tuple(sorted({tag.recipe for tag in tags}))
    first = tags[0]
    match level:
        case FineLevel():
            return FineLabel(dataset=first.dataset, recipe=first.recipe)
        case MediumLevel():
            return MediumLabel(dataset=first.dataset, mixture_of=mixture)
        case GlobalLevel():
            return GlobalLabel(mixture_of=mixture)


@dataclass(frozen=True, slots=True)
class SystemCounts:
    """How many distinct systems, and distinct matrices, a report group measured."""

    n_systems: int
    n_matrices: int


def system_counts(frame: pl.DataFrame, level: ReportLevel) -> dict[GroupLabel, SystemCounts]:
    """Distinct measured systems and matrices of each group (a matrix is per dataset)."""
    keys = level_columns(level)
    counts = {
        "n_systems": pl.col(SYSTEM_ID).n_unique(),
        "n_matrices": pl.struct(["dataset", "matrix_id"]).n_unique(),
    }
    counted = (
        frame.group_by(keys, maintain_order=True).agg(**counts) if keys else frame.select(**counts)
    )
    mixtures = mixtures_by_key(frame, keys)
    return {
        group_label(level, row, mixtures): SystemCounts(
            n_systems=row["n_systems"], n_matrices=row["n_matrices"]
        )
        for row in counted.iter_rows(named=True)
    }


def level_columns(level: ReportLevel) -> list[str]:
    match level:
        case FineLevel():
            return ["dataset", "recipe"]
        case MediumLevel():
            return ["dataset"]
        case GlobalLevel():
            return []


def mixtures_by_key(frame: pl.DataFrame, keys: list[str]) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Recipes pooled by each group of ``keys``, sorted."""
    if not keys:
        return {(): tuple(sorted(frame["recipe"].unique()))}
    pooled = frame.group_by(keys, maintain_order=True).agg(mixture=pl.col("recipe").unique().sort())
    return {
        tuple(row[key] for key in keys): tuple(row["mixture"])
        for row in pooled.iter_rows(named=True)
    }


def group_label(
    level: ReportLevel,
    row: Mapping[str, object],
    mixtures: Mapping[tuple[str, ...], tuple[str, ...]],
) -> GroupLabel:
    match level:
        case FineLevel():
            return FineLabel(dataset=DatasetId(str(row["dataset"])), recipe=str(row["recipe"]))
        case MediumLevel():
            dataset = str(row["dataset"])
            return MediumLabel(dataset=DatasetId(dataset), mixture_of=mixtures[(dataset,)])
        case GlobalLevel():
            return GlobalLabel(mixture_of=mixtures[()])
