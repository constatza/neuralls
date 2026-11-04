"""Typed figure plans: the scientific context every figure is drawn and recorded with."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import polars as pl

from neuralls.domain.benchmarking.paired import PAIRED_SCHEMA, SPEED_RATIO_SCHEMA
from neuralls.domain.benchmarking.profiles import PROFILE_SERIES_SCHEMA
from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.presentation.methods import ConfiguredMethodPresentation
from neuralls.domain.solver.models.figure_scope import FigureScope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.cohorts import (
    COHORT_RATIO_OF_SUMS_SCHEMA,
    COHORT_SUMMARY_SCHEMA,
)
from neuralls.domain.solver.reporting.matrix_breakdown import MATRIX_BREAKDOWN_SCHEMA
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.domain.solver.reporting.summary import SUMMARY_SCHEMA, Population
from neuralls.platform.reporting.benchmark_events import METHOD_ARTIFACT_DIR
from neuralls.platform.reporting.figure_evidence import (
    FigureInterval,
    FigureTimingEvidence,
    MethodOutcomeCounts,
    SetupSharing,
)
from neuralls.shared.rhs_recipe import RhsRecipe
from neuralls.shared.scientific_description.policy import RenderProfile
from neuralls.shared.types import DatasetId, PreconditionerName, RunId

CARD_PROFILE = RenderProfile(name="card", include_context=True)
"""Figure cards store the scope's full text on one line; wrapping is a drawing concern."""


class FigureKind(StrEnum):
    AGGREGATE_BARS = "aggregate_bars"
    ONLINE_TIME = "online_time"
    SOLVE_TIME = "solve_time"
    PAIRED_TIME = "paired_time"
    PERFORMANCE_PROFILE = "performance_profile"
    ECDF_ITERATIONS = "ecdf_iterations"
    ECDF_TIME = "ecdf_time"
    SPEED_RATIO = "speed_ratio"
    ITERATION_MEANS = "iteration_means"
    TIME_MEANS = "time_means"
    SETUP_MEANS = "setup_means"
    TIME_PER_ITERATION = "time_per_iteration"
    ITERATION_DISTRIBUTION = "iteration_distribution"
    TIME_DISTRIBUTION = "time_distribution"
    MATRIX_HEATMAP = "matrix_heatmap"
    RESIDUAL_HISTORY = "residual_history"
    ENERGY_HISTORY = "energy_history"


GROUP_FIGURE_KINDS: frozenset[FigureKind] = frozenset(FigureKind) - {
    FigureKind.RESIDUAL_HISTORY,
    FigureKind.ENERGY_HISTORY,
}
"""Kinds drawn from a group outcome; each has a compact plotted table beside it."""

TABLE_SUFFIX = ".parquet"
"""A plotted table shares its figure's stem and directory; only the suffix differs."""


class TableKind(StrEnum):
    """The closed set of tables a group figure writes beside itself, each the frame it draws.

    ``PAIRED`` holds the per-system pairs of a scatter, ``SUMMARY`` one row of statistics per
    method, ``SERIES`` the points of a step curve, ``SPEED_RATIO`` one row of ratio statistics per
    compared method and ``WORKLOAD_TOTALS`` one row of workload sums per method.
    ``COHORT_SUMMARY`` and ``COHORT_RATIO_OF_SUMS`` hold per-method statistics by cohort: one block
    per recipe, and at a mixture level the pooled block too (``domain/solver/reporting/cohorts.py``).
    ``MATRIX_BREAKDOWN`` holds one row per method and matrix, exactly the cells of a heatmap.
    """

    SUMMARY = "summary"
    SERIES = "series"
    PAIRED = "paired"
    SPEED_RATIO = "speed_ratio"
    WORKLOAD_TOTALS = "workload_totals"
    COHORT_SUMMARY = "cohort_summary"
    COHORT_RATIO_OF_SUMS = "cohort_ratio_of_sums"
    MATRIX_BREAKDOWN = "matrix_breakdown"


type TableSchema = Mapping[str, pl.DataType | type[pl.DataType]]

WORKLOAD_TOTALS_SCHEMA: TableSchema = {
    "method_id": pl.String,
    "n_planned_rhs": pl.Int64,
    "n_attempted_rhs": pl.Int64,
    "n_converged": pl.Int64,
    "allocated_setup_s": pl.Float64,
    "solve_total_s": pl.Float64,
    "workload_online_s": pl.Float64,
}
"""One row per method: workload totals, summed over its attempted systems."""

TABLE_SCHEMAS: Mapping[TableKind, TableSchema] = {
    TableKind.SUMMARY: SUMMARY_SCHEMA,
    TableKind.SERIES: PROFILE_SERIES_SCHEMA,
    TableKind.PAIRED: PAIRED_SCHEMA,
    TableKind.SPEED_RATIO: SPEED_RATIO_SCHEMA,
    TableKind.WORKLOAD_TOTALS: WORKLOAD_TOTALS_SCHEMA,
    TableKind.COHORT_SUMMARY: COHORT_SUMMARY_SCHEMA,
    TableKind.COHORT_RATIO_OF_SUMS: COHORT_RATIO_OF_SUMS_SCHEMA,
    TableKind.MATRIX_BREAKDOWN: MATRIX_BREAKDOWN_SCHEMA,
}
"""The columns and dtypes, in order, of the table each kind stores; the frames come from the
domain statistics, so a schema change there reaches the stored tables through this mapping."""


@dataclass(frozen=True, slots=True)
class SiblingTable:
    """The figure has a table of the given kind with the same stem in the same directory."""

    name: str
    kind: TableKind


@dataclass(frozen=True, slots=True)
class NoTable:
    """The figure draws per-iteration histories, which stay in the compare artifacts."""


type FigureTable = SiblingTable | NoTable


class Statistic(StrEnum):
    """What a group figure draws of its metric.

    The values of the bar statistics are also the summary-table columns their bars are read from.
    ``QUANTILES`` draws a distribution whose box and whiskers are the summary's ``q05`` to ``q95``;
    ``MEAN_PER_MATRIX`` draws one mean per method and matrix.
    """

    CONVERGED_COUNT = "converged_count"
    WORKLOAD_TOTAL = "workload_total"
    ECDF = "ecdf"
    PERFORMANCE_PROFILE = "performance_profile"
    PAIRED_VALUES = "paired_values"
    SPEED_RATIO_CDF = "speed_ratio_cdf"
    MEAN_OVER_SYSTEMS = "mean_over_systems"
    MEAN_OVER_MATRICES = "mean_over_matrices"
    RATIO_OF_SUMS = "ratio_of_sums"
    QUANTILES = "quantiles"
    MEAN_PER_MATRIX = "mean_per_matrix"


class FailureTreatment(StrEnum):
    CENSORED = "censored"
    OMITTED = "omitted"
    NONE = "none"


class ReferenceSelectionKind(StrEnum):
    CONFIGURED = "configured"
    NONE = "none"
    UNAVAILABLE = "unavailable"


class TimingPolicy(StrEnum):
    """How a figure's timing numbers treat one-off setup against per-RHS solves."""

    SETUP_PER_MATRIX_PLUS_SOLVE_PER_RHS = "setup once per dataset/matrix + solve per planned RHS"
    DIAGNOSTIC_EXCLUDED_FROM_ONLINE = "diagnostic histories; excluded from online timing"


class GroupAggregationRule(StrEnum):
    """How a group figure combines the systems of its population into one number."""

    PER_PLANNED_SYSTEM = "one observation per planned (matrix, RHS) system"
    WORKLOAD_SUM = (
        "sum observed setup and solve event costs; recipe contributions follow planned RHS "
        "membership"
    )


class MethodAggregationRule(StrEnum):
    """Which convergence histories a method or system figure draws."""

    PER_ATTEMPTED_RHS = "one diagnostic history per attempted RHS"
    PER_METHOD_ON_SYSTEM = "one diagnostic history per method on this system"


@dataclass(frozen=True, slots=True)
class FigureMethod:
    """One method a figure draws: its identity, scientific presentation and config digest."""

    method_id: PreconditionerName
    presentation: ConfiguredMethodPresentation
    config_digest: str


@dataclass(frozen=True, slots=True)
class FigureSource:
    """The stored artifacts of one method run that a figure was computed from."""

    run_id: RunId
    artifact_paths: tuple[str, ...]


def method_source(run_id: RunId, *artifacts: Path) -> FigureSource:
    """The source of a figure: ``artifacts`` of the method workload logged in ``run_id``."""
    return FigureSource(
        run_id=run_id,
        artifact_paths=tuple(
            f"{METHOD_ARTIFACT_DIR}/{artifact.as_posix()}" for artifact in artifacts
        ),
    )


@dataclass(frozen=True, slots=True)
class _FigureContext:
    """What every figure records about the population, methods and sources it was drawn from."""

    kind: FigureKind
    metric_definition: str
    report_level: ReportLevelName
    scope: FigureScope
    dataset_ids: tuple[DatasetId, ...]
    recipes: tuple[RhsRecipe, ...]
    n_matrices: int
    n_planned_rhs: int
    reference_selection: ReferenceSelection
    timing_policy: TimingPolicy
    methods: tuple[FigureMethod, ...]
    sources: tuple[FigureSource, ...]
    recipe_counts: tuple[int, ...]
    recipe_weights: tuple[float, ...]
    solver_contexts: tuple[SolverTimingContext, ...]


@dataclass(frozen=True, slots=True)
class GroupFigureMetadata(_FigureContext):
    """Context of a figure aggregated over a group of systems.

    ``metric`` is the per-system quantity the figure summarizes, ``population`` the systems it
    is taken over and ``statistic`` what is drawn of it. ``failure_treatment`` says what the
    figure does with failed systems. The remaining fields record the evidence under the numbers:
    the ``interval`` drawn, the right-hand-side counts the metric divides setup over, how many
    systems each method has in each outcome, and how reliable the times are.
    """

    metric: MetricId
    population: Population
    statistic: Statistic
    failure_treatment: FailureTreatment
    aggregation_rule: GroupAggregationRule
    interval: FigureInterval
    setup_sharing: SetupSharing
    outcome_counts: tuple[MethodOutcomeCounts, ...]
    timing: FigureTimingEvidence

    def __post_init__(self) -> None:
        """Require a kind drawn from a group outcome."""
        if self.kind not in GROUP_FIGURE_KINDS:
            raise ValueError(f"{self.kind.value} is not a group figure kind")


@dataclass(frozen=True, slots=True)
class MethodFigureMetadata(_FigureContext):
    """Context of a convergence-history figure of one method or one system.

    Histories are diagnostics, so they summarize no metric and failed systems are drawn
    as they ended.
    """

    aggregation_rule: MethodAggregationRule

    def __post_init__(self) -> None:
        """Require a kind drawn from stored method histories."""
        if self.kind in GROUP_FIGURE_KINDS:
            raise ValueError(f"{self.kind.value} is not a method figure kind")


type FigureMetadata = GroupFigureMetadata | MethodFigureMetadata
"""The closed set of figure contexts; a card stores exactly one of them."""


@dataclass(frozen=True, slots=True)
class FigurePlan[M: FigureMetadata]:
    """All presentation and scientific context decided before drawing.

    ``title`` is fixed editorial wording; the subtitle is ``metadata.scope`` rendered
    when the figure is drawn.
    """

    filename: str
    title: str
    metadata: M


def collision_key(filename: str) -> str:
    """The case-folded safe spelling of ``filename``; equal keys name the same artifact."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).lower()


@dataclass(frozen=True, slots=True)
class StagedFigure:
    """A drawn figure file together with the plan that describes it."""

    plan: FigurePlan
    path: Path

    def __post_init__(self) -> None:
        """Require the file to carry the planned name."""
        if self.path.name != self.plan.filename:
            raise ValueError(
                f"staged file {self.path.name!r} does not carry the planned name "
                f"{self.plan.filename!r}"
            )


@dataclass(frozen=True, slots=True)
class StagedTabledFigure:
    """A staged group figure together with the table it plots, uploaded beside it."""

    figure: StagedFigure
    table: pl.DataFrame


type UploadableFigure = StagedFigure | StagedTabledFigure
"""A figure without a plotted table, or one with it; the card says which one is correct."""


MAX_SLUG_LENGTH = 80


def text_slug(text: str) -> str:
    """Turn rendered scientific text into a safe, human-readable file stem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return stem[:MAX_SLUG_LENGTH].rstrip("_") or "figure"


@dataclass(slots=True)
class FigureSlugs:
    """Assigns each report subject one file stem derived from its rendered text.

    Subjects whose text collapses to the same stem get a numeric suffix in
    encounter order, so names stay unique and deterministic without exposing the
    machine id that keys the subject.
    """

    _by_subject: dict[str, str] = field(default_factory=dict)

    def slug(self, subject_key: str, rendered_text: str) -> str:
        """Return the stable stem of ``subject_key``, allocating it on first use."""
        existing = self._by_subject.get(subject_key)
        if existing is not None:
            return existing
        base = text_slug(rendered_text)
        taken = {value.lower() for value in self._by_subject.values()}
        candidate, index = base, 1
        while candidate.lower() in taken:
            index += 1
            suffix = f"_{index}"
            candidate = f"{base[: MAX_SLUG_LENGTH - len(suffix)].rstrip('_')}{suffix}"
        self._by_subject[subject_key] = candidate
        return candidate
