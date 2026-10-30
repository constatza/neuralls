"""Typed figure plans: the scientific context every figure is drawn and recorded with."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import polars as pl

from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.presentation.methods import ConfiguredMethodPresentation
from neuralls.domain.solver.models.figure_scope import FigureScope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.benchmark_events import METHOD_ARTIFACT_DIR
from neuralls.shared.rhs_recipe import RhsRecipe
from neuralls.shared.scientific_description.policy import RenderProfile
from neuralls.shared.types import PreconditionerName

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
    RESIDUAL_HISTORY = "residual_history"
    ENERGY_HISTORY = "energy_history"


GROUP_FIGURE_KINDS: frozenset[FigureKind] = frozenset(FigureKind) - {
    FigureKind.RESIDUAL_HISTORY,
    FigureKind.ENERGY_HISTORY,
}
"""Kinds drawn from a group outcome; each has a compact plotted table beside it."""

TABLE_SUFFIX = ".parquet"
"""A plotted table shares its figure's stem and directory; only the suffix differs."""

GROUP_TABLE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method_id": pl.String,
    "n_planned_rhs": pl.Int64,
    "n_attempted_rhs": pl.Int64,
    "n_converged": pl.Int64,
    "setup_total_s": pl.Float64,
    "solve_total_s": pl.Float64,
    "workload_online_s": pl.Float64,
}
"""The one schema of the plotted table: one row per method drawn, workload totals only."""


@dataclass(frozen=True, slots=True)
class SiblingTable:
    """The figure has a plotted table with the same stem in the same directory."""

    name: str


@dataclass(frozen=True, slots=True)
class NoTable:
    """The figure draws per-iteration histories, which stay in the compare artifacts."""


type FigureTable = SiblingTable | NoTable


def table_of(figure: str, kind: FigureKind) -> FigureTable:
    """The plotted table of ``figure`` of ``kind``; the kind alone decides whether one exists."""
    if kind not in GROUP_FIGURE_KINDS:
        return NoTable()
    return SiblingTable(f"{figure.removesuffix('.png')}{TABLE_SUFFIX}")


class TimingScenario(StrEnum):
    SOLVE = "solve"
    ONLINE = "online"


class FailureTreatment(StrEnum):
    CENSORED = "censored"
    OMITTED = "omitted"
    NONE = "none"


class ReferenceSelectionKind(StrEnum):
    CONFIGURED = "configured"
    NONE = "none"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FigureMethod:
    """One method a figure draws: its identity, scientific presentation and config digest."""

    method_id: PreconditionerName
    presentation: ConfiguredMethodPresentation
    config_digest: str


@dataclass(frozen=True, slots=True)
class FigureSource:
    """The stored artifacts of one method run that a figure was computed from."""

    run_id: str
    artifact_paths: tuple[str, ...]


def method_source(run_id: str, *artifacts: Path) -> FigureSource:
    """The source of a figure: ``artifacts`` of the method workload logged in ``run_id``."""
    return FigureSource(
        run_id=run_id,
        artifact_paths=tuple(
            f"{METHOD_ARTIFACT_DIR}/{artifact.as_posix()}" for artifact in artifacts
        ),
    )


@dataclass(frozen=True, slots=True)
class FigureMetadata:
    kind: FigureKind
    metric_definition: str
    scenario: TimingScenario | None
    report_level: ReportLevelName
    scope: FigureScope
    dataset_ids: tuple[str, ...]
    recipes: tuple[RhsRecipe, ...]
    n_matrices: int
    n_planned_rhs: int
    reference_selection: ReferenceSelection
    failure_treatment: FailureTreatment
    timing_policy: str
    methods: tuple[FigureMethod, ...]
    sources: tuple[FigureSource, ...]
    recipe_counts: tuple[int, ...] = ()
    aggregation_rule: str = ""
    recipe_weights: tuple[float, ...] = ()
    solver_contexts: tuple[SolverTimingContext, ...] = ()


@dataclass(frozen=True, slots=True)
class FigurePlan:
    """All presentation and scientific context decided before drawing.

    ``title`` is fixed editorial wording; the subtitle is ``metadata.scope`` rendered
    when the figure is drawn.
    """

    filename: str
    title: str
    metadata: FigureMetadata


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


def sanitize_filename(value: str) -> str:
    """Create the stable slug used by figure plans and collision checks."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").lower()


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


def make_plan(
    kind: FigureKind,
    metadata: FigureMetadata,
    *,
    group_slug: str,
) -> FigurePlan:
    titles = {
        FigureKind.AGGREGATE_BARS: "Convergence",
        FigureKind.ONLINE_TIME: "Online Time",
        FigureKind.SOLVE_TIME: "Solve Time",
        FigureKind.PAIRED_TIME: "Online Time",
        FigureKind.PERFORMANCE_PROFILE: "Performance Profile",
        FigureKind.ECDF_ITERATIONS: "Iterations",
        FigureKind.ECDF_TIME: (
            "Solve Time" if metadata.scenario is TimingScenario.SOLVE else "Online Time"
        ),
        FigureKind.SPEED_RATIO: "Speed Ratio",
        FigureKind.RESIDUAL_HISTORY: "Residual History",
        FigureKind.ENERGY_HISTORY: "Energy History",
    }
    scenario = f"_{metadata.scenario.value}" if metadata.scenario else ""
    filename = f"{kind.value}_{sanitize_filename(metadata.report_level.value)}_{sanitize_filename(group_slug)}{scenario}.png"
    return FigurePlan(filename, titles[kind], metadata)
