"""Long-form metrics table over comparison results, and its three report levels.

Every number a report shows is "this preconditioner, over these systems". The
preconditioner is therefore the defining key of every level and never a level
of its own; a level only chooses which *other* columns also key the groups.
Missing measurements are nulls, which polars skips in means and spreads, so an
unmeasured solve is never silently read as 0.0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    FineLabel,
    GlobalLabel,
    GroupLabel,
    MediumLabel,
    ProjectedComparisonResult,
    SetupSharingPolicy,
)
from neuralls.shared.types import CostProvenance, DatasetId, MatrixId

PRECONDITIONER = "preconditioner"
SYSTEM_ID = "system_id"

METRICS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "dataset": pl.String,
    "recipe": pl.String,
    "matrix_id": pl.Int64,
    "rhs_ordinal": pl.Int64,
    SYSTEM_ID: pl.String,
    PRECONDITIONER: pl.String,
    "setup_s": pl.Float64,
    "generation_s": pl.Float64,
    "solve_s": pl.Float64,
    "iterations": pl.Int64,
    "time_to_solution_s": pl.Float64,
    "peak_memory_bytes": pl.Int64,
}


@dataclass(frozen=True, slots=True)
class SystemTag:
    """Where one solved system sits in the case: dataset, RHS recipe, matrix, RHS.

    Attributes:
        dataset: The dataset the system's matrix and RHS come from.
        recipe: Rendered RHS recipe label (e.g. ``"sinusoidal ω=6.28"``).
        matrix_id: Physical matrix index within the dataset.
        rhs_ordinal: Position of this RHS among the matrix's RHS draws.
        system_id: Stable identifier of the system, used to key per-system points.
    """

    dataset: DatasetId
    recipe: str
    matrix_id: MatrixId
    rhs_ordinal: int
    system_id: str


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


def build_metrics_frame(
    results_by_system: Mapping[SystemTag, Mapping[str, CGComparisonResult]],
) -> pl.DataFrame:
    """One row per (system, preconditioner).

    ``generation_s`` is null when the generation cost is unavailable (the same
    results ``time_to_solution_s`` treats as incomparable, which is null too);
    ``solve_s`` is null when the solve was not measured.
    """
    rows = [
        {
            "dataset": tag.dataset,
            "recipe": tag.recipe,
            "matrix_id": tag.matrix_id,
            "rhs_ordinal": tag.rhs_ordinal,
            SYSTEM_ID: tag.system_id,
            PRECONDITIONER: name,
            "setup_s": result.setup_cost,
            "generation_s": _available_generation_seconds(result),
            "solve_s": result.solve_time_seconds,
            "iterations": result.iterations,
            "time_to_solution_s": time_to_solution(result),
            "peak_memory_bytes": result.peak_memory_bytes,
        }
        for tag, system_results in results_by_system.items()
        for name, result in system_results.items()
    ]
    return pl.DataFrame(rows, schema=METRICS_SCHEMA)


def report_level(
    frame: pl.DataFrame,
    level: ReportLevel,
    sharing_by_name: Mapping[str, SetupSharingPolicy],
) -> dict[GroupLabel, dict[str, ProjectedComparisonResult]]:
    """Reduce the frame to one projection per (group, preconditioner).

    Args:
        frame: Output of `build_metrics_frame`.
        level: Which columns, besides the preconditioner, key the groups.
        sharing_by_name: Each preconditioner's setup/generation sharing policy.

    Returns:
        Projections per group label, groups and preconditioners in first-seen order.

    Raises:
        KeyError: If a preconditioner in the frame has no sharing policy.
    """
    keys = _keys(level)
    aggregated = frame.group_by([*keys, PRECONDITIONER], maintain_order=True).agg(
        n_systems=pl.len(),
        setup_total=pl.col("setup_s").sum(),
        generation_mean=pl.col("generation_s").mean(),
        solve_mean=pl.col("solve_s").mean(),
        solve_std=pl.col("solve_s").std(),
        iterations_total=pl.col("iterations").sum(),
        iterations_std=pl.col("iterations").std(),
    )
    mixtures = _mixtures(frame, keys)
    report: dict[GroupLabel, dict[str, ProjectedComparisonResult]] = {}
    for row in aggregated.iter_rows(named=True):
        label = _label(level, row, mixtures)
        n = row["n_systems"]
        generation_mean = row["generation_mean"]
        solve_mean = row["solve_mean"]
        name = row[PRECONDITIONER]
        report.setdefault(label, {})[name] = ProjectedComparisonResult(
            preconditioner_name=name,
            n_systems=n,
            setup_sharing=sharing_by_name[name],
            setup_cost_total_seconds=row["setup_total"],
            # Mean over the available results scaled to n, so the per-system
            # average (total / n) is that mean rather than diluted by
            # unavailable results.
            generation_cost_total_seconds=(
                None if generation_mean is None else generation_mean * n
            ),
            total_solve_time_seconds=0.0 if solve_mean is None else solve_mean * n,
            total_iterations=row["iterations_total"],
            solve_time_std_seconds=row["solve_std"],
            iterations_std=row["iterations_std"],
        )
    return report


def time_to_solution_by_system(
    frame: pl.DataFrame, level: ReportLevel
) -> dict[GroupLabel, dict[str, dict[str, float]]]:
    """Each group's per-preconditioner, per-system time-to-solution.

    Systems whose time is incomparable (null) are omitted, in frame order.
    """
    keys = _keys(level)
    mixtures = _mixtures(frame, keys)
    comparable = frame.filter(pl.col("time_to_solution_s").is_not_null())
    out: dict[GroupLabel, dict[str, dict[str, float]]] = {}
    for row in comparable.iter_rows(named=True):
        label = _label(level, row, mixtures)
        out.setdefault(label, {}).setdefault(row[PRECONDITIONER], {})[row[SYSTEM_ID]] = row[
            "time_to_solution_s"
        ]
    return out


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
    keys = _keys(level)
    counts = {
        "n_systems": pl.col(SYSTEM_ID).n_unique(),
        "n_matrices": pl.struct(["dataset", "matrix_id"]).n_unique(),
    }
    counted = (
        frame.group_by(keys, maintain_order=True).agg(**counts) if keys else frame.select(**counts)
    )
    mixtures = _mixtures(frame, keys)
    return {
        _label(level, row, mixtures): SystemCounts(
            n_systems=row["n_systems"], n_matrices=row["n_matrices"]
        )
        for row in counted.iter_rows(named=True)
    }


def _keys(level: ReportLevel) -> list[str]:
    match level:
        case FineLevel():
            return ["dataset", "recipe"]
        case MediumLevel():
            return ["dataset"]
        case GlobalLevel():
            return []


def _mixtures(frame: pl.DataFrame, keys: list[str]) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Recipes pooled by each group of ``keys``, sorted."""
    if not keys:
        return {(): tuple(sorted(frame["recipe"].unique()))}
    pooled = frame.group_by(keys, maintain_order=True).agg(mixture=pl.col("recipe").unique().sort())
    return {
        tuple(row[key] for key in keys): tuple(row["mixture"])
        for row in pooled.iter_rows(named=True)
    }


def _label(
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


def _available_generation_seconds(result: CGComparisonResult) -> float | None:
    cost = result.generation_cost
    if cost is None or cost.provenance is CostProvenance.UNAVAILABLE:
        return None
    return cost.wall_time_seconds


def time_to_solution(result: CGComparisonResult) -> float | None:
    """Total time, or `None` when an unavailable cost makes it incomparable."""
    return None if result.has_unavailable_cost else result.total_time_seconds
