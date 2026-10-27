"""Typed figure plans and the session-level figure manifest."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import tomli_w

from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceSelection,
    ReferenceUnavailable,
)
from neuralls.domain.solver.models.figure_scope import FigureScope, render_figure_scope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.platform.storage.facts_codec import encode_figure_scope
from neuralls.shared.rhs_recipe import RhsRecipe, recipe_key
from neuralls.shared.scientific_description.policy import (
    NotationPolicy,
    RenderProfile,
    RenderTarget,
)

MANIFEST_PROFILE = RenderProfile(name="manifest", include_context=True)
"""Figure manifests store the scope's full text on one line; wrapping is a drawing concern."""


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
class FigureMetadata:
    kind: FigureKind
    metric_definition: str
    scenario: TimingScenario | None
    report_level: str
    scope: FigureScope
    dataset_ids: tuple[str, ...]
    recipes: tuple[RhsRecipe, ...]
    n_matrices: int
    n_planned_rhs: int
    reference_selection: ReferenceSelection | ReferenceSelectionKind
    failure_treatment: FailureTreatment
    timing_policy: str
    recipe_counts: tuple[int, ...] = ()
    aggregation_rule: str = ""
    recipe_weights: tuple[float, ...] = ()
    solver_contexts: tuple[SolverTimingContext, ...] = ()
    reference_reason: str | None = None
    sources: tuple[str, ...] = (
        "setup_events.parquet",
        "solve_attempts.parquet",
        "summary.toml",
    )


@dataclass(frozen=True, slots=True)
class FigurePlan:
    """All presentation and scientific context decided before drawing.

    ``title`` is fixed editorial wording; the subtitle is ``metadata.scope`` rendered
    when the figure is drawn.
    """

    filename: str
    title: str
    metadata: FigureMetadata


def _collision_key(filename: str) -> str:
    stem = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).lower()


@dataclass(frozen=True, slots=True)
class FigureManifest:
    entries: dict[str, FigureMetadata] = field(default_factory=dict)

    def add(self, filename: str, metadata: FigureMetadata) -> FigureManifest:
        """Return a manifest containing ``filename``; reject duplicate safe names."""
        if not filename or Path(filename).name != filename:
            raise ValueError("figure filename must be a non-empty basename")
        key = _collision_key(filename)
        if any(_collision_key(existing) == key for existing in self.entries):
            raise ValueError(f"duplicate or colliding figure filename: {filename}")
        self.entries[filename] = metadata
        return self

    def add_plan(self, plan: FigurePlan) -> FigureManifest:
        """Register a pre-draw plan after its filename has been reserved."""
        return self.add(plan.filename, plan.metadata)

    def validate_plans(self, plans: Iterable[FigurePlan]) -> None:
        """Reject all planned collisions before any artifact is uploaded."""
        names: dict[str, str] = {}
        for plan in plans:
            key = _collision_key(plan.filename)
            if key in names or any(_collision_key(existing) == key for existing in self.entries):
                raise ValueError(f"duplicate or colliding figure filename: {plan.filename}")
            names[key] = plan.filename

    def as_dict(self, notation: NotationPolicy) -> dict[str, object]:
        """The manifest as TOML-ready data; scopes carry structure and their rendered text."""
        plain = replace(notation, target=RenderTarget.PLAIN)
        figures: dict[str, object] = {}
        for filename in sorted(self.entries):
            metadata = self.entries[filename]
            reference = metadata.reference_selection
            reference_method_id: str | None = None
            if isinstance(reference, ReferenceSelectionKind):
                reference_name = reference.value
            elif isinstance(reference, ConfiguredReference):
                reference_name = ReferenceSelectionKind.CONFIGURED.value
                reference_method_id = str(reference.name)
            elif isinstance(reference, NoReferenceConfigured):
                reference_name = ReferenceSelectionKind.NONE.value
            elif isinstance(reference, ReferenceUnavailable):
                reference_name = ReferenceSelectionKind.UNAVAILABLE.value
            else:
                raise TypeError(f"unsupported reference selection: {reference!r}")
            entry: dict[str, object] = {
                "kind": metadata.kind.value,
                "metric_definition": metadata.metric_definition,
                "report_level": metadata.report_level,
                "scope": {
                    **encode_figure_scope(metadata.scope),
                    "display": render_figure_scope(metadata.scope, MANIFEST_PROFILE, plain).text,
                },
                "dataset_ids": list(metadata.dataset_ids),
                "recipes": [recipe_key(recipe) for recipe in metadata.recipes],
                "n_matrices": metadata.n_matrices,
                "n_planned_rhs": metadata.n_planned_rhs,
                "reference_selection": reference_name,
                "failure_treatment": metadata.failure_treatment.value,
                "timing_policy": metadata.timing_policy,
                "aggregation_rule": metadata.aggregation_rule,
                "recipe_counts": list(metadata.recipe_counts),
                "recipe_weights": list(metadata.recipe_weights),
                "solver_contexts": [asdict(context) for context in metadata.solver_contexts],
                "sources": list(metadata.sources),
            }
            if reference_method_id is not None:
                entry["reference_method_id"] = reference_method_id
            if metadata.reference_reason is not None:
                entry["reference_reason"] = metadata.reference_reason
            if metadata.scenario is not None:
                entry["scenario"] = metadata.scenario.value
            figures[filename] = entry
        return {"figures": figures}


REQUIRED_MANIFEST_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        "kind",
        "metric_definition",
        "report_level",
        "scope",
        "dataset_ids",
        "recipes",
        "n_matrices",
        "n_planned_rhs",
        "reference_selection",
        "failure_treatment",
        "timing_policy",
        "aggregation_rule",
        "recipe_counts",
        "recipe_weights",
        "solver_contexts",
        "sources",
    }
)


class InvalidFigureManifestError(ValueError):
    """A payload is not a complete figure manifest."""


def validate_figure_manifest(payload: Mapping[str, object]) -> None:
    """Raise unless ``payload`` has a ``figures`` table of complete entries."""
    figures = payload.get("figures")
    if not isinstance(figures, Mapping):
        raise InvalidFigureManifestError("figure manifest has no 'figures' table")
    for filename, entry in figures.items():
        if not isinstance(entry, Mapping):
            raise InvalidFigureManifestError(f"figure entry {filename!r} is not a table")
        missing = sorted(REQUIRED_MANIFEST_ENTRY_FIELDS - entry.keys())
        if missing:
            raise InvalidFigureManifestError(
                f"figure entry {filename!r} is missing fields: {', '.join(missing)}"
            )
        scope = entry["scope"]
        if not isinstance(scope, Mapping) or not {"kind", "display"} <= scope.keys():
            raise InvalidFigureManifestError(
                f"figure entry {filename!r} scope must be a table with 'kind' and 'display'"
            )


def write_figure_manifest(
    path: str | Path, manifest: FigureManifest, notation: NotationPolicy
) -> None:
    Path(path).write_bytes(tomli_w.dumps(manifest.as_dict(notation)).encode("utf-8"))


def sanitize_filename(value: str) -> str:
    """Create the stable slug used by figure plans and collision checks."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").lower()


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
    filename = f"{kind.value}_{sanitize_filename(metadata.report_level)}_{sanitize_filename(group_slug)}{scenario}.png"
    return FigurePlan(filename, titles[kind], metadata)
