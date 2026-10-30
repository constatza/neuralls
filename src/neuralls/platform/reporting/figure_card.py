"""The figure card: one TOML document beside each figure holding what produced it.

A card is the single record of a figure's parameters (solver tolerances, methods drawn,
failure treatment, population) and of the stored artifacts its numbers came from. The
figure index only lists cards; it never repeats their parameters.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, assert_never

import tomli_w

from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceSelection,
    ReferenceUnavailable,
)
from neuralls.domain.solver.models.figure_scope import render_figure_scope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.figure_metadata import (
    CARD_PROFILE,
    FailureTreatment,
    FigureKind,
    FigureMetadata,
    FigureMethod,
    FigureSource,
    FigureTable,
    NoTable,
    ReferenceSelectionKind,
    SiblingTable,
    TimingScenario,
    table_of,
)
from neuralls.platform.storage.facts_codec import (
    decode_figure_scope,
    decode_method_presentation,
    decode_recipe,
    encode_figure_scope,
    encode_method_presentation,
    encode_recipe,
)
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import PreconditionerName

CARD_SUFFIX = ".toml"
"""A card shares its figure's stem and directory; only the suffix differs."""

FIGURE_SUFFIX = ".png"

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "figure",
        "kind",
        "title",
        "metric_definition",
        "level",
        "failure_treatment",
        "aggregation_rule",
        "timing_policy",
        "scope",
        "methods",
        "counts",
        "reference",
        "parameters",
        "source",
    }
)
_REQUIRED_COUNTS: frozenset[str] = frozenset(
    {"n_matrices", "n_planned_rhs", "dataset_ids", "recipes", "recipe_counts", "recipe_weights"}
)
_REQUIRED_METHOD_FIELDS: frozenset[str] = frozenset({"method_id", "config_digest", "presentation"})
_REQUIRED_SOURCE_FIELDS: frozenset[str] = frozenset({"session_run_id", "methods"})
_REQUIRED_SOURCE_METHOD_FIELDS: frozenset[str] = frozenset({"run_id", "artifact_paths"})


class InvalidFigureCardError(ValueError):
    """A payload is not a complete figure card."""


@dataclass(frozen=True, slots=True)
class FigureCard:
    """Everything recorded about one figure, validated at construction.

    Attributes:
        figure: File name of the PNG the card describes.
        title: The figure's fixed editorial title.
        scope_display: ``metadata.scope`` rendered as plain text.
        session_run_id: The compare session the numbers were read from.
        metadata: The scientific context the figure was drawn with; its ``sources`` name the
            method runs and artifacts the figure was computed from.
    """

    figure: str
    title: str
    scope_display: str
    session_run_id: str
    metadata: FigureMetadata

    def __post_init__(self) -> None:
        """Require a PNG basename and a session run id."""
        if not self.figure or Path(self.figure).name != self.figure:
            raise ValueError("figure card must name a file by its basename")
        if not self.figure.endswith(FIGURE_SUFFIX):
            raise ValueError(f"figure card must describe a {FIGURE_SUFFIX} file: {self.figure}")
        if not self.session_run_id:
            raise ValueError("figure card needs the session run id its numbers came from")

    @property
    def table(self) -> FigureTable:
        """The plotted table beside the figure, decided by the figure's kind alone."""
        return table_of(self.figure, self.metadata.kind)


def build_figure_card(
    figure: str,
    title: str,
    metadata: FigureMetadata,
    *,
    session_run_id: str,
    notation: NotationPolicy,
) -> FigureCard:
    """The card of ``figure``, rendering its scope as plain text."""
    plain = replace(notation, target=RenderTarget.PLAIN)
    return FigureCard(
        figure=figure,
        title=title,
        scope_display=render_figure_scope(metadata.scope, CARD_PROFILE, plain).text,
        session_run_id=session_run_id,
        metadata=metadata,
    )


def card_name(figure: str) -> str:
    """The file name of the card that sits beside ``figure``."""
    return f"{figure.removesuffix(FIGURE_SUFFIX)}{CARD_SUFFIX}"


def _encode_reference(reference: ReferenceSelection) -> dict[str, Any]:
    match reference:
        case ConfiguredReference(name=name):
            return {"selection": ReferenceSelectionKind.CONFIGURED.value, "method_id": str(name)}
        case NoReferenceConfigured():
            return {"selection": ReferenceSelectionKind.NONE.value}
        case ReferenceUnavailable(name=name, reason=reason):
            return {
                "selection": ReferenceSelectionKind.UNAVAILABLE.value,
                "method_id": str(name),
                "reason": reason,
            }
        case unreachable:
            assert_never(unreachable)


def _decode_reference(payload: Mapping[str, Any]) -> ReferenceSelection:
    match ReferenceSelectionKind(payload["selection"]):
        case ReferenceSelectionKind.CONFIGURED:
            return ConfiguredReference(PreconditionerName(payload["method_id"]))
        case ReferenceSelectionKind.NONE:
            return NoReferenceConfigured()
        case ReferenceSelectionKind.UNAVAILABLE:
            return ReferenceUnavailable(
                PreconditionerName(payload["method_id"]), str(payload["reason"])
            )


def card_payload(card: FigureCard) -> dict[str, Any]:
    """The card as TOML-ready data: structure, typed values and the rendered scope text."""
    metadata = card.metadata
    payload: dict[str, Any] = {
        "figure": card.figure,
        "kind": metadata.kind.value,
        "title": card.title,
        "metric_definition": metadata.metric_definition,
        "level": metadata.report_level.value,
        "failure_treatment": metadata.failure_treatment.value,
        "aggregation_rule": metadata.aggregation_rule,
        "timing_policy": metadata.timing_policy,
        "scope": {**encode_figure_scope(metadata.scope), "display": card.scope_display},
        "methods": [
            {
                "method_id": str(method.method_id),
                "config_digest": method.config_digest,
                "presentation": encode_method_presentation(method.presentation),
            }
            for method in metadata.methods
        ],
        "counts": {
            "n_matrices": metadata.n_matrices,
            "n_planned_rhs": metadata.n_planned_rhs,
            "dataset_ids": list(metadata.dataset_ids),
            "recipes": [encode_recipe(recipe) for recipe in metadata.recipes],
            "recipe_counts": list(metadata.recipe_counts),
            "recipe_weights": list(metadata.recipe_weights),
        },
        "reference": _encode_reference(metadata.reference_selection),
        "parameters": [asdict(context) for context in metadata.solver_contexts],
        "source": {
            "session_run_id": card.session_run_id,
            "methods": [
                {"run_id": source.run_id, "artifact_paths": list(source.artifact_paths)}
                for source in metadata.sources
            ],
        },
    }
    if metadata.scenario is not None:
        payload["scenario"] = metadata.scenario.value
    match card.table:
        case SiblingTable(name=name):
            payload["table"] = name
        case NoTable():
            pass
        case unreachable:
            assert_never(unreachable)
    return payload


def _require(table: object, fields: frozenset[str], what: str) -> Mapping[str, Any]:
    if not isinstance(table, Mapping):
        raise InvalidFigureCardError(f"figure card {what} is not a table")
    missing = sorted(fields - table.keys())
    if missing:
        raise InvalidFigureCardError(f"figure card {what} is missing fields: {', '.join(missing)}")
    return table


def _require_tables(entries: object, fields: frozenset[str], what: str) -> list[Mapping[str, Any]]:
    if not isinstance(entries, list):
        raise InvalidFigureCardError(f"figure card {what} is not an array of tables")
    return [_require(entry, fields, what) for entry in entries]


def validate_figure_card(payload: Mapping[str, Any]) -> None:
    """Raise unless ``payload`` is a complete figure card."""
    card = _require(payload, _REQUIRED_FIELDS, "document")
    _require(card["scope"], frozenset({"kind", "display"}), "scope")
    _require_tables(card["methods"], _REQUIRED_METHOD_FIELDS, "method")
    _require(card["counts"], _REQUIRED_COUNTS, "counts")
    _require(card["reference"], frozenset({"selection"}), "reference")
    if not isinstance(card["parameters"], list):
        raise InvalidFigureCardError("figure card parameters is not an array of tables")
    source = _require(card["source"], _REQUIRED_SOURCE_FIELDS, "source")
    _require_tables(source["methods"], _REQUIRED_SOURCE_METHOD_FIELDS, "source method")


def decode_figure_card(payload: Mapping[str, Any]) -> FigureCard:
    """Validate ``payload`` and rebuild the card it encodes."""
    validate_figure_card(payload)
    counts = payload["counts"]
    scenario = payload.get("scenario")
    metadata = FigureMetadata(
        kind=FigureKind(payload["kind"]),
        metric_definition=str(payload["metric_definition"]),
        scenario=None if scenario is None else TimingScenario(scenario),
        report_level=ReportLevelName(payload["level"]),
        scope=decode_figure_scope(payload["scope"]),
        dataset_ids=tuple(str(dataset) for dataset in counts["dataset_ids"]),
        recipes=tuple(decode_recipe(recipe) for recipe in counts["recipes"]),
        n_matrices=int(counts["n_matrices"]),
        n_planned_rhs=int(counts["n_planned_rhs"]),
        reference_selection=_decode_reference(payload["reference"]),
        failure_treatment=FailureTreatment(payload["failure_treatment"]),
        timing_policy=str(payload["timing_policy"]),
        methods=tuple(
            FigureMethod(
                method_id=PreconditionerName(method["method_id"]),
                presentation=decode_method_presentation(method["presentation"]),
                config_digest=str(method["config_digest"]),
            )
            for method in payload["methods"]
        ),
        sources=tuple(
            FigureSource(
                run_id=str(source["run_id"]),
                artifact_paths=tuple(str(artifact) for artifact in source["artifact_paths"]),
            )
            for source in payload["source"]["methods"]
        ),
        recipe_counts=tuple(int(count) for count in counts["recipe_counts"]),
        aggregation_rule=str(payload["aggregation_rule"]),
        recipe_weights=tuple(float(weight) for weight in counts["recipe_weights"]),
        solver_contexts=tuple(SolverTimingContext(**context) for context in payload["parameters"]),
    )
    card = FigureCard(
        figure=str(payload["figure"]),
        title=str(payload["title"]),
        scope_display=str(payload["scope"]["display"]),
        session_run_id=str(payload["source"]["session_run_id"]),
        metadata=metadata,
    )
    _check_table(card, payload.get("table"))
    return card


def _check_table(card: FigureCard, recorded: object) -> None:
    """Require the card's ``table`` entry to be exactly the table its kind implies."""
    match card.table:
        case SiblingTable(name=name):
            if recorded != name:
                raise InvalidFigureCardError(f"figure card table must be {name}, not {recorded!r}")
        case NoTable():
            if recorded is not None:
                raise InvalidFigureCardError(f"figure card of {card.metadata.kind} has no table")
        case unreachable:
            assert_never(unreachable)


def write_figure_card(path: Path, card: FigureCard) -> None:
    """Write ``card`` as TOML; the output is a deterministic function of the card."""
    path.write_bytes(tomli_w.dumps(card_payload(card)).encode("utf-8"))


def read_figure_card(path: Path) -> FigureCard:
    """Read and validate the card stored at ``path``."""
    return decode_figure_card(tomllib.loads(path.read_text(encoding="utf-8")))
