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
from neuralls.domain.benchmarking.uncertainty import MatrixClusterBootstrap
from neuralls.domain.solver.models.figure_scope import render_figure_scope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.reporting.figure_evidence import (
    SINGLE_REPEAT_WARNING,
    FigureInterval,
    FigureTimingEvidence,
    MethodOutcomeCounts,
    MethodTimingSpread,
    NoInterval,
    NoSharedSetup,
    NoTimingEvidence,
    SetupSharedOverRhs,
    SetupSharing,
    TimingEvidence,
    TimingRepeats,
)
from neuralls.platform.reporting.figure_metadata import (
    CARD_PROFILE,
    GROUP_FIGURE_KINDS,
    FailureTreatment,
    FigureKind,
    FigureMetadata,
    FigureMethod,
    FigureSource,
    FigureTable,
    GroupAggregationRule,
    GroupFigureMetadata,
    MethodAggregationRule,
    MethodFigureMetadata,
    NoTable,
    ReferenceSelectionKind,
    SiblingTable,
    Statistic,
    TimingPolicy,
)
from neuralls.platform.reporting.figure_planning import table_of
from neuralls.platform.storage.facts_codec import (
    decode_figure_scope,
    decode_method_presentation,
    decode_recipe,
    encode_figure_scope,
    encode_method_presentation,
    encode_recipe,
)
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import DatasetId, PreconditionerName, RunId, SessionRunId

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
    session_run_id: SessionRunId
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
    session_run_id: SessionRunId,
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


def _encode_interval(interval: FigureInterval) -> dict[str, Any]:
    match interval:
        case NoInterval():
            return {}
        case MatrixClusterBootstrap(matrices=matrices, level=level, resamples=resamples):
            return {
                "interval": {
                    "method": MatrixClusterBootstrap.NAME,
                    "level": level,
                    "resamples": resamples,
                    "matrices": matrices,
                }
            }
        case unreachable:
            assert_never(unreachable)


def _decode_interval(payload: Mapping[str, Any]) -> FigureInterval:
    if "interval" not in payload:
        return NoInterval()
    interval = _require(
        payload["interval"], frozenset({"method", "level", "resamples", "matrices"}), "interval"
    )
    if interval["method"] != MatrixClusterBootstrap.NAME:
        raise InvalidFigureCardError(
            f"figure card interval method is unknown: {interval['method']}"
        )
    return MatrixClusterBootstrap(
        matrices=int(interval["matrices"]),
        level=float(interval["level"]),
        resamples=int(interval["resamples"]),
    )


def _encode_setup_sharing(sharing: SetupSharing) -> dict[str, Any]:
    match sharing:
        case NoSharedSetup():
            return {}
        case SetupSharedOverRhs(rhs_counts=rhs_counts):
            return {"r_m": list(rhs_counts)}
        case unreachable:
            assert_never(unreachable)


def _decode_setup_sharing(payload: Mapping[str, Any]) -> SetupSharing:
    if "r_m" not in payload:
        return NoSharedSetup()
    return SetupSharedOverRhs(tuple(int(count) for count in payload["r_m"]))


def _encode_timing(timing: FigureTimingEvidence) -> dict[str, Any]:
    match timing:
        case NoTimingEvidence():
            return {}
        case TimingEvidence(spreads=spreads, repeats=repeats):
            table = {str(spread.method): spread.median_spread for spread in spreads}
            match repeats:
                case TimingRepeats.REPEATED:
                    return {"timing_spread": table}
                case TimingRepeats.SINGLE:
                    return {"timing_spread": table, "timing_warning": SINGLE_REPEAT_WARNING}
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)


def _decode_timing(payload: Mapping[str, Any]) -> FigureTimingEvidence:
    if "timing_spread" not in payload:
        if "timing_warning" in payload:
            raise InvalidFigureCardError("figure card has a timing warning without timing spreads")
        return NoTimingEvidence()
    spreads = tuple(
        MethodTimingSpread(PreconditionerName(method), float(spread))
        for method, spread in payload["timing_spread"].items()
    )
    repeats = TimingRepeats.SINGLE if "timing_warning" in payload else TimingRepeats.REPEATED
    return TimingEvidence(spreads, repeats)


def _group_payload(metadata: GroupFigureMetadata) -> dict[str, Any]:
    """The fields only a group figure records: what it summarizes and the evidence under it."""
    return {
        "failure_treatment": metadata.failure_treatment.value,
        "metric": metadata.metric.value,
        "population": metadata.population.value,
        "statistic": metadata.statistic.value,
        "outcome_counts": {
            str(counts.method): {outcome.value: n for outcome, n in counts.counts.items()}
            for counts in metadata.outcome_counts
        },
        **_encode_interval(metadata.interval),
        **_encode_setup_sharing(metadata.setup_sharing),
        **_encode_timing(metadata.timing),
    }


def card_payload(card: FigureCard) -> dict[str, Any]:
    """The card as TOML-ready data: structure, typed values and the rendered scope text."""
    metadata = card.metadata
    payload: dict[str, Any] = {
        "figure": card.figure,
        "kind": metadata.kind.value,
        "title": card.title,
        "metric_definition": metadata.metric_definition,
        "level": metadata.report_level.value,
        "aggregation_rule": metadata.aggregation_rule.value,
        "timing_policy": metadata.timing_policy.value,
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
            "dataset_ids": [str(dataset) for dataset in metadata.dataset_ids],
            "recipes": [encode_recipe(recipe) for recipe in metadata.recipes],
            "recipe_counts": list(metadata.recipe_counts),
            "recipe_weights": list(metadata.recipe_weights),
        },
        "reference": _encode_reference(metadata.reference_selection),
        "parameters": [asdict(context) for context in metadata.solver_contexts],
        "source": {
            "session_run_id": card.session_run_id,
            "methods": [
                {"run_id": str(source.run_id), "artifact_paths": list(source.artifact_paths)}
                for source in metadata.sources
            ],
        },
    }
    match metadata:
        case GroupFigureMetadata():
            payload.update(_group_payload(metadata))
        case MethodFigureMetadata():
            pass
        case unreachable:
            assert_never(unreachable)
    match card.table:
        case SiblingTable(name=name, kind=kind):
            payload["table"] = name
            payload["table_kind"] = kind.value
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


def _decode_group_metadata(kind: FigureKind, payload: Mapping[str, Any]) -> GroupFigureMetadata:
    _require(
        payload,
        frozenset({"failure_treatment", "metric", "population", "statistic", "outcome_counts"}),
        "group document",
    )
    return GroupFigureMetadata(
        **_common_fields(kind, payload),
        interval=_decode_interval(payload),
        setup_sharing=_decode_setup_sharing(payload),
        outcome_counts=tuple(
            MethodOutcomeCounts(
                PreconditionerName(method),
                {Outcome(outcome): int(n) for outcome, n in counts.items()},
            )
            for method, counts in payload["outcome_counts"].items()
        ),
        timing=_decode_timing(payload),
        metric=MetricId(payload["metric"]),
        population=Population(payload["population"]),
        statistic=Statistic(payload["statistic"]),
        failure_treatment=FailureTreatment(payload["failure_treatment"]),
        aggregation_rule=GroupAggregationRule(payload["aggregation_rule"]),
    )


def _decode_method_metadata(kind: FigureKind, payload: Mapping[str, Any]) -> MethodFigureMetadata:
    return MethodFigureMetadata(
        **_common_fields(kind, payload),
        aggregation_rule=MethodAggregationRule(payload["aggregation_rule"]),
    )


def _common_fields(kind: FigureKind, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of every figure context, decoded from a validated card payload."""
    counts = payload["counts"]
    return {
        "kind": kind,
        "metric_definition": str(payload["metric_definition"]),
        "report_level": ReportLevelName(payload["level"]),
        "scope": decode_figure_scope(payload["scope"]),
        "dataset_ids": tuple(DatasetId(str(dataset)) for dataset in counts["dataset_ids"]),
        "recipes": tuple(decode_recipe(recipe) for recipe in counts["recipes"]),
        "n_matrices": int(counts["n_matrices"]),
        "n_planned_rhs": int(counts["n_planned_rhs"]),
        "reference_selection": _decode_reference(payload["reference"]),
        "timing_policy": TimingPolicy(payload["timing_policy"]),
        "methods": tuple(
            FigureMethod(
                method_id=PreconditionerName(method["method_id"]),
                presentation=decode_method_presentation(method["presentation"]),
                config_digest=str(method["config_digest"]),
            )
            for method in payload["methods"]
        ),
        "sources": tuple(
            FigureSource(
                run_id=RunId(str(source["run_id"])),
                artifact_paths=tuple(str(artifact) for artifact in source["artifact_paths"]),
            )
            for source in payload["source"]["methods"]
        ),
        "recipe_counts": tuple(int(count) for count in counts["recipe_counts"]),
        "recipe_weights": tuple(float(weight) for weight in counts["recipe_weights"]),
        "solver_contexts": tuple(
            SolverTimingContext(**context) for context in payload["parameters"]
        ),
    }


def decode_figure_card(payload: Mapping[str, Any]) -> FigureCard:
    """Validate ``payload`` and rebuild the card it encodes."""
    validate_figure_card(payload)
    kind = FigureKind(payload["kind"])
    metadata = (
        _decode_group_metadata(kind, payload)
        if kind in GROUP_FIGURE_KINDS
        else _decode_method_metadata(kind, payload)
    )
    card = FigureCard(
        figure=str(payload["figure"]),
        title=str(payload["title"]),
        scope_display=str(payload["scope"]["display"]),
        session_run_id=SessionRunId(str(payload["source"]["session_run_id"])),
        metadata=metadata,
    )
    _check_table(card, payload)
    return card


def _check_table(card: FigureCard, payload: Mapping[str, Any]) -> None:
    """Require the card's table entries to be exactly the table its kind implies."""
    match card.table:
        case SiblingTable(name=name, kind=kind):
            if payload.get("table") != name:
                raise InvalidFigureCardError(
                    f"figure card table must be {name}, not {payload.get('table')!r}"
                )
            if payload.get("table_kind") != kind.value:
                raise InvalidFigureCardError(
                    f"figure card table kind must be {kind.value}, "
                    f"not {payload.get('table_kind')!r}"
                )
        case NoTable():
            if "table" in payload or "table_kind" in payload:
                raise InvalidFigureCardError(f"figure card of {card.metadata.kind} has no table")
        case unreachable:
            assert_never(unreachable)


def write_figure_card(path: Path, card: FigureCard) -> None:
    """Write ``card`` as TOML; the output is a deterministic function of the card."""
    path.write_bytes(tomli_w.dumps(card_payload(card)).encode("utf-8"))


def read_figure_card(path: Path) -> FigureCard:
    """Read and validate the card stored at ``path``."""
    return decode_figure_card(tomllib.loads(path.read_text(encoding="utf-8")))
