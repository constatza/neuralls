"""The figure card: one TOML document beside each figure holding what produced it.

A card is the single record of a figure's parameters (solver tolerances, methods drawn,
failure treatment, population) and of the stored artifacts its numbers came from. The
figure index only lists cards; it never repeats their parameters.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, assert_never

import tomli_w

from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceSelection,
    ReferenceUnavailable,
)
from neuralls.domain.benchmarking.uncertainty import (
    MIN_INTERVAL_MATRICES,
    BootstrapKind,
    ClusterBootstrap,
    TooFewMatrices,
    cluster_bootstrap,
)
from neuralls.domain.presentation.methods import ConfiguredMethodPresentation
from neuralls.domain.solver.models.figure_scope import render_figure_scope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.cohorts import (
    Cohort,
    MixtureCohort,
    WholeGroup,
    mixture_label,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.reporting.figure_evidence import (
    SINGLE_REPEAT_WARNING,
    FigureInterval,
    FigureTimingEvidence,
    MethodInterval,
    MethodOutcomeCounts,
    MethodTimingSpread,
    NoSharedSetup,
    NotDrawnByKind,
    NoTimingEvidence,
    PairInterval,
    PerMethodIntervals,
    PerPairIntervals,
    SetupSharedOverRhs,
    SetupSharing,
    TimingEvidence,
    TimingRepeats,
)
from neuralls.platform.reporting.figure_formats import figure_files
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
from neuralls.shared.errors import ReasonedError
from neuralls.shared.rhs_recipe import RhsRecipe, recipe_key
from neuralls.shared.runtime_environment import (
    RuntimeEnvironment,
    decode_environment,
    encode_environment,
)
from neuralls.shared.scientific_description.catalog import SEED
from neuralls.shared.scientific_description.models import IntegerFact
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.system_scaling import RhsScaling, parse_system_scaling
from neuralls.shared.types import DatasetId, PreconditionerName, RunId, SessionRunId

CARD_SUFFIX = ".toml"
"""A card shares its figure's stem and directory; only the suffix differs."""

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "stem",
        "files",
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
_REQUIRED_SOURCE_FIELDS: frozenset[str] = frozenset({"session_run_id", "environment", "methods"})
_REQUIRED_SOURCE_METHOD_FIELDS: frozenset[str] = frozenset({"run_id", "artifact_paths"})


class InvalidFigureCardReason(StrEnum):
    """How a payload fails to be a complete figure card."""

    INTERVAL_INVALID = "interval_invalid"
    INTERVAL_METHOD_NOT_A_STRING = "interval_method_not_a_string"
    INTERVAL_MINIMUM_WRONG = "interval_minimum_wrong"
    INTERVAL_RESAMPLED_WRONG = "interval_resampled_wrong"
    INTERVAL_CONTRADICTS_MATRIX_COUNT = "interval_contradicts_matrix_count"
    INTERVAL_KIND_UNKNOWN = "interval_kind_unknown"
    INTERVAL_FIELDS_NOT_EXACT = "interval_fields_not_exact"
    INTERVAL_FIELD_WRONG_TYPE = "interval_field_wrong_type"
    COHORT_NOT_THE_MIXTURE = "cohort_not_the_mixture"
    TIMING_WARNING_WITHOUT_SPREADS = "timing_warning_without_spreads"
    NOT_A_TABLE = "not_a_table"
    MISSING_FIELDS = "missing_fields"
    NOT_AN_ARRAY_OF_TABLES = "not_an_array_of_tables"
    PARAMETERS_NOT_AN_ARRAY = "parameters_not_an_array"
    FILES_MISMATCH = "files_mismatch"
    TABLE_NAME_MISMATCH = "table_name_mismatch"
    TABLE_KIND_MISMATCH = "table_kind_mismatch"
    UNEXPECTED_TABLE = "unexpected_table"


class InvalidFigureCardError(ReasonedError[InvalidFigureCardReason], ValueError):
    """A payload is not a complete figure card."""


class FigureCardFieldsMissingError(InvalidFigureCardError):
    """A figure card table lacks required fields.

    Attributes:
        missing: The names that are absent, in the order the message lists them.
    """

    def __init__(self, detail: str, missing: tuple[str, ...]) -> None:
        super().__init__(InvalidFigureCardReason.MISSING_FIELDS, detail)
        self.missing = missing

    def _constructor_args(self) -> tuple[object, ...]:
        return (self.detail, self.missing)


@dataclass(frozen=True, slots=True)
class FigureCard:
    """Everything recorded about one figure, validated at construction.

    Attributes:
        stem: File stem shared by the figure's files, one per figure format.
        title: The figure's fixed editorial title.
        scope_display: ``metadata.scope`` rendered as plain text.
        session_run_id: The compare session the numbers were read from.
        metadata: The scientific context the figure was drawn with; its ``sources`` name the
            method runs and artifacts the figure was computed from.
        environment: The device, CPU, threads, library versions and code revisions of the
            session that produced the timings, read from the session run.
    """

    stem: str
    title: str
    scope_display: str
    session_run_id: SessionRunId
    metadata: FigureMetadata
    environment: RuntimeEnvironment

    def __post_init__(self) -> None:
        """Require a file stem and a session run id."""
        if not self.stem or Path(self.stem).name != self.stem:
            raise ValueError("figure card must name its files by a stem without directories")
        if not self.session_run_id:
            raise ValueError("figure card needs the session run id its numbers came from")

    @property
    def files(self) -> tuple[str, ...]:
        """The figure's files, one per figure format."""
        return figure_files(self.stem)

    @property
    def table(self) -> FigureTable:
        """The plotted table beside the figure, decided by the figure's kind alone."""
        return table_of(self.stem, self.metadata.kind)


def build_figure_card(
    stem: str,
    title: str,
    metadata: FigureMetadata,
    *,
    session_run_id: SessionRunId,
    environment: RuntimeEnvironment,
    notation: NotationPolicy,
) -> FigureCard:
    """The card of the figure with file stem ``stem``, rendering its scope as plain text."""
    plain = replace(notation, target=RenderTarget.PLAIN)
    return FigureCard(
        stem=stem,
        title=title,
        scope_display=render_figure_scope(metadata.scope, CARD_PROFILE, plain).text,
        session_run_id=session_run_id,
        metadata=metadata,
        environment=environment,
    )


def card_name(stem: str) -> str:
    """The file name of the card that sits beside the figure with file stem ``stem``."""
    return f"{stem}{CARD_SUFFIX}"


def _encode_solver_context(context: SolverTimingContext) -> dict[str, Any]:
    """One ``[[parameters]]`` table: the scalings become a TOML table and a string."""
    return {
        **asdict(context),
        "system_scaling": context.system_scaling.model_dump(mode="json"),
        "rhs_scaling": context.rhs_scaling.value,
    }


def _decode_solver_context(payload: Mapping[str, Any]) -> SolverTimingContext:
    fields: dict[str, Any] = dict(payload)
    fields["system_scaling"] = parse_system_scaling(payload["system_scaling"])
    fields["rhs_scaling"] = RhsScaling(payload["rhs_scaling"])
    return SolverTimingContext(**fields)


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


class FigureIntervalKind(StrEnum):
    """The card's name for whether a figure draws intervals."""

    PER_METHOD = "per_method"
    PER_METHOD_PAIR = "per_method_pair"
    NOT_DRAWN_BY_KIND = "not_drawn_by_kind"


class MethodIntervalKind(StrEnum):
    """The card's name for one method's interval, or the reason it has none."""

    MATRIX_CLUSTER_BOOTSTRAP = BootstrapKind.MATRIX_CLUSTER.value
    TWO_LEVEL_BOOTSTRAP = BootstrapKind.TWO_LEVEL.value
    TOO_FEW_MATRICES = "too_few_matrices"


_BOOTSTRAP_FIELDS = frozenset({"method", "kind", "resampled", "matrices", "level", "resamples"})
_METHOD_INTERVAL_FIELDS: Mapping[MethodIntervalKind, frozenset[str]] = {
    MethodIntervalKind.TOO_FEW_MATRICES: frozenset({"method", "kind", "matrices", "minimum"}),
    MethodIntervalKind.MATRIX_CLUSTER_BOOTSTRAP: _BOOTSTRAP_FIELDS,
    MethodIntervalKind.TWO_LEVEL_BOOTSTRAP: _BOOTSTRAP_FIELDS,
}
"""The exact keys of a method's entry in ``[[interval.methods]]``, by its kind."""


_REFERENCE_METHOD = "reference_method"
"""The compared method of a ``[[interval.pairs]]`` entry, beside a method entry's fields."""


def _encode_interval(interval: FigureInterval) -> dict[str, Any]:
    match interval:
        case NotDrawnByKind():
            return {"kind": FigureIntervalKind.NOT_DRAWN_BY_KIND.value}
        case PerMethodIntervals(methods=methods):
            return {
                "kind": FigureIntervalKind.PER_METHOD.value,
                "methods": [_encode_method_interval(entry) for entry in methods],
            }
        case PerPairIntervals(pairs=pairs):
            return {
                "kind": FigureIntervalKind.PER_METHOD_PAIR.value,
                "pairs": [
                    _encode_method_interval(MethodInterval(entry.method, entry.interval))
                    | {_REFERENCE_METHOD: entry.reference_method}
                    for entry in pairs
                ],
            }
        case unreachable:
            assert_never(unreachable)


def _encode_method_interval(entry: MethodInterval) -> dict[str, Any]:
    match entry.interval:
        case TooFewMatrices(matrices=matrices):
            return {
                "method": entry.method,
                "kind": MethodIntervalKind.TOO_FEW_MATRICES.value,
                "matrices": matrices,
                "minimum": MIN_INTERVAL_MATRICES,
            }
        case ClusterBootstrap(kind=kind, matrices=matrices, level=level, resamples=resamples):
            return {
                "method": entry.method,
                "kind": kind.value,
                "resampled": kind.description,
                "matrices": matrices,
                "level": level,
                "resamples": resamples,
            }
        case unreachable:
            assert_never(unreachable)


def _decode_interval(payload: object) -> FigureInterval:
    interval = _require(payload, frozenset({"kind"}), "interval")
    match _kind(FigureIntervalKind, interval):
        case FigureIntervalKind.NOT_DRAWN_BY_KIND:
            _exact(interval, frozenset({"kind"}))
            return NotDrawnByKind()
        case FigureIntervalKind.PER_METHOD:
            _exact(interval, frozenset({"kind", "methods"}))
            entries = _require_tables(interval["methods"], frozenset({"kind"}), "interval method")
            return _checked(
                lambda: PerMethodIntervals(tuple(_decode_method_interval(e) for e in entries))
            )
        case FigureIntervalKind.PER_METHOD_PAIR:
            _exact(interval, frozenset({"kind", "pairs"}))
            entries = _require_tables(
                interval["pairs"], frozenset({"kind", _REFERENCE_METHOD}), "interval pair"
            )
            return _checked(
                lambda: PerPairIntervals(tuple(_decode_pair_interval(e) for e in entries))
            )
        case unreachable:
            assert_never(unreachable)


def _checked[I: (PerMethodIntervals, PerPairIntervals)](decode: Callable[[], I]) -> I:
    """``decode()``, its interval types' own invariants (a level above one) as card errors."""
    try:
        return decode()
    except InvalidFigureCardError:
        raise
    except ValueError as error:
        raise InvalidFigureCardError(
            InvalidFigureCardReason.INTERVAL_INVALID, f"figure card interval: {error}"
        ) from error


def _decode_pair_interval(entry: Mapping[str, Any]) -> PairInterval:
    reference = entry[_REFERENCE_METHOD]
    if not isinstance(reference, str):
        raise InvalidFigureCardError(
            InvalidFigureCardReason.INTERVAL_METHOD_NOT_A_STRING,
            "figure card interval reference method is not a string",
        )
    method = _decode_method_interval(
        {key: value for key, value in entry.items() if key != _REFERENCE_METHOD}
    )
    return PairInterval(method.method, PreconditionerName(reference), method.interval)


def _decode_method_interval(entry: Mapping[str, Any]) -> MethodInterval:
    kind = _kind(MethodIntervalKind, entry)
    _exact(entry, _METHOD_INTERVAL_FIELDS[kind])
    method = entry["method"]
    if not isinstance(method, str):
        raise InvalidFigureCardError(
            InvalidFigureCardReason.INTERVAL_METHOD_NOT_A_STRING,
            "figure card interval method is not a string",
        )
    matrices = _typed(entry, "matrices", int)
    match kind:
        case MethodIntervalKind.TOO_FEW_MATRICES:
            bootstrap = cluster_bootstrap(matrices, BootstrapKind.MATRIX_CLUSTER)
        case MethodIntervalKind.MATRIX_CLUSTER_BOOTSTRAP | MethodIntervalKind.TWO_LEVEL_BOOTSTRAP:
            bootstrap = cluster_bootstrap(matrices, BootstrapKind(kind.value))
            if entry["resampled"] != BootstrapKind(kind.value).description:
                raise InvalidFigureCardError(
                    InvalidFigureCardReason.INTERVAL_RESAMPLED_WRONG,
                    f"figure card interval of {method} resamples {entry['resampled']!r}, "
                    f"not what {kind.value} resamples",
                )
        case unreachable:
            assert_never(unreachable)
    match kind, bootstrap:
        case MethodIntervalKind.TOO_FEW_MATRICES, TooFewMatrices():
            if _typed(entry, "minimum", int) != MIN_INTERVAL_MATRICES:
                raise InvalidFigureCardError(
                    InvalidFigureCardReason.INTERVAL_MINIMUM_WRONG,
                    f"figure card interval minimum is not {MIN_INTERVAL_MATRICES}",
                )
            return MethodInterval(PreconditionerName(method), bootstrap)
        case (
            MethodIntervalKind.MATRIX_CLUSTER_BOOTSTRAP | MethodIntervalKind.TWO_LEVEL_BOOTSTRAP,
            ClusterBootstrap(),
        ):
            drawn = replace(
                bootstrap,
                level=_typed(entry, "level", float),
                resamples=_typed(entry, "resamples", int),
            )
            return MethodInterval(PreconditionerName(method), drawn)
        case _:
            raise InvalidFigureCardError(
                InvalidFigureCardReason.INTERVAL_CONTRADICTS_MATRIX_COUNT,
                f"figure card interval of {method} is {kind.value}, which its matrix count "
                "contradicts",
            )


def _kind[K: (FigureIntervalKind, MethodIntervalKind)](
    enum: type[K], table: Mapping[str, Any]
) -> K:
    try:
        return enum(table["kind"])
    except ValueError as error:
        raise InvalidFigureCardError(
            InvalidFigureCardReason.INTERVAL_KIND_UNKNOWN,
            f"figure card interval kind is unknown: {table['kind']}",
        ) from error


def _exact(table: Mapping[str, Any], fields: frozenset[str]) -> None:
    if set(table) != fields:
        raise InvalidFigureCardError(
            InvalidFigureCardReason.INTERVAL_FIELDS_NOT_EXACT,
            f"figure card interval must have exactly the fields {', '.join(sorted(fields))}",
        )


def _typed[T: (int, float)](table: Mapping[str, Any], key: str, kind: type[T]) -> T:
    """``table[key]``, which TOML must have parsed as exactly ``kind`` (a bool is no int)."""
    value = table[key]
    if type(value) is not kind:
        raise InvalidFigureCardError(
            InvalidFigureCardReason.INTERVAL_FIELD_WRONG_TYPE,
            f"figure card interval {key} is not a {kind.__name__}",
        )
    return value


def _encode_cohort(cohort: Cohort, recipes: tuple[RhsRecipe, ...]) -> dict[str, Any]:
    """A figure pooled over a mixture names itself by its recipes' keys; a class figure needs none."""
    match cohort:
        case WholeGroup():
            return {}
        case MixtureCohort():
            return {"cohort": _mixture_label(recipes)}
        case unreachable:
            assert_never(unreachable)


def _mixture_label(recipes: tuple[RhsRecipe, ...]) -> str:
    return mixture_label(recipe_key(recipe) for recipe in recipes)


def _decode_cohort(payload: Mapping[str, Any], recipes: tuple[RhsRecipe, ...]) -> Cohort:
    match payload.get("cohort"):
        case None:
            return WholeGroup()
        case str(label) if label == _mixture_label(recipes):
            return MixtureCohort()
        case other:
            raise InvalidFigureCardError(
                InvalidFigureCardReason.COHORT_NOT_THE_MIXTURE,
                f"figure card cohort is not the mixture of its recipes: {other!r}",
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
            raise InvalidFigureCardError(
                InvalidFigureCardReason.TIMING_WARNING_WITHOUT_SPREADS,
                "figure card has a timing warning without timing spreads",
            )
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
        **_encode_cohort(metadata.cohort, metadata.recipes),
        "metric": metadata.metric.value,
        "population": metadata.population.value,
        "statistic": metadata.statistic.value,
        "outcome_counts": {
            str(counts.method): {outcome.value: n for outcome, n in counts.counts.items()}
            for counts in metadata.outcome_counts
        },
        "interval": _encode_interval(metadata.interval),
        **_encode_setup_sharing(metadata.setup_sharing),
        **_encode_timing(metadata.timing),
        "excluded_zero_best": metadata.excluded_zero_best,
    }


def card_payload(card: FigureCard) -> dict[str, Any]:
    """The card as TOML-ready data: structure, typed values and the rendered scope text."""
    metadata = card.metadata
    payload: dict[str, Any] = {
        "stem": card.stem,
        "files": list(card.files),
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
                **_setup_seed(method.presentation),
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
        "parameters": [_encode_solver_context(context) for context in metadata.solver_contexts],
        "source": {
            "session_run_id": card.session_run_id,
            "environment": encode_environment(card.environment),
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


def _setup_seed(presentation: ConfiguredMethodPresentation) -> dict[str, str]:
    """How a seeded method's setups were seeded: per matrix, from its configured seed.

    The configured seed stays the method's identity and fact; each matrix's setup is built with
    a seed derived from it and the matrix (``setup_config_for_matrix``), which the card states so
    the fact is not read as the seed of every setup.
    """
    description = presentation.method
    for fact in (*description.facts, *description.context_facts):
        match fact:
            case IntegerFact(key=key, value=seed) if key == SEED:
                return {"setup_seed": f"derived per matrix from seed s = {seed}"}
            case _:
                continue
    return {}


def _require(table: object, fields: frozenset[str], what: str) -> Mapping[str, Any]:
    if not isinstance(table, Mapping):
        raise InvalidFigureCardError(
            InvalidFigureCardReason.NOT_A_TABLE, f"figure card {what} is not a table"
        )
    missing = sorted(fields - table.keys())
    if missing:
        raise FigureCardFieldsMissingError(
            f"figure card {what} is missing fields: {', '.join(missing)}", tuple(missing)
        )
    return table


def _require_tables(entries: object, fields: frozenset[str], what: str) -> list[Mapping[str, Any]]:
    if not isinstance(entries, list):
        raise InvalidFigureCardError(
            InvalidFigureCardReason.NOT_AN_ARRAY_OF_TABLES,
            f"figure card {what} is not an array of tables",
        )
    return [_require(entry, fields, what) for entry in entries]


def validate_figure_card(payload: Mapping[str, Any]) -> None:
    """Raise unless ``payload`` is a complete figure card."""
    card = _require(payload, _REQUIRED_FIELDS, "document")
    _require(card["scope"], frozenset({"kind", "display"}), "scope")
    _require_tables(card["methods"], _REQUIRED_METHOD_FIELDS, "method")
    _require(card["counts"], _REQUIRED_COUNTS, "counts")
    _require(card["reference"], frozenset({"selection"}), "reference")
    if not isinstance(card["parameters"], list):
        raise InvalidFigureCardError(
            InvalidFigureCardReason.PARAMETERS_NOT_AN_ARRAY,
            "figure card parameters is not an array of tables",
        )
    source = _require(card["source"], _REQUIRED_SOURCE_FIELDS, "source")
    _require_tables(source["methods"], _REQUIRED_SOURCE_METHOD_FIELDS, "source method")


def _decode_group_metadata(kind: FigureKind, payload: Mapping[str, Any]) -> GroupFigureMetadata:
    _require(
        payload,
        frozenset(
            {
                "failure_treatment",
                "metric",
                "population",
                "statistic",
                "outcome_counts",
                "excluded_zero_best",
                "interval",
            }
        ),
        "group document",
    )
    common = _common_fields(kind, payload)
    return GroupFigureMetadata(
        **common,
        cohort=_decode_cohort(payload, common["recipes"]),
        interval=_decode_interval(payload["interval"]),
        setup_sharing=_decode_setup_sharing(payload),
        outcome_counts=tuple(
            MethodOutcomeCounts(
                PreconditionerName(method),
                {Outcome(outcome): int(n) for outcome, n in counts.items()},
            )
            for method, counts in payload["outcome_counts"].items()
        ),
        timing=_decode_timing(payload),
        excluded_zero_best=int(payload["excluded_zero_best"]),
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
            _decode_solver_context(context) for context in payload["parameters"]
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
        stem=str(payload["stem"]),
        title=str(payload["title"]),
        scope_display=str(payload["scope"]["display"]),
        session_run_id=SessionRunId(str(payload["source"]["session_run_id"])),
        metadata=metadata,
        environment=decode_environment(
            {key: str(value) for key, value in payload["source"]["environment"].items()}
        ),
    )
    if tuple(payload["files"]) != card.files:
        raise InvalidFigureCardError(
            InvalidFigureCardReason.FILES_MISMATCH,
            f"figure card files must be {list(card.files)}, not {payload['files']!r}",
        )
    _check_table(card, payload)
    return card


def _check_table(card: FigureCard, payload: Mapping[str, Any]) -> None:
    """Require the card's table entries to be exactly the table its kind implies."""
    match card.table:
        case SiblingTable(name=name, kind=kind):
            if payload.get("table") != name:
                raise InvalidFigureCardError(
                    InvalidFigureCardReason.TABLE_NAME_MISMATCH,
                    f"figure card table must be {name}, not {payload.get('table')!r}",
                )
            if payload.get("table_kind") != kind.value:
                raise InvalidFigureCardError(
                    InvalidFigureCardReason.TABLE_KIND_MISMATCH,
                    f"figure card table kind must be {kind.value}, "
                    f"not {payload.get('table_kind')!r}",
                )
        case NoTable():
            if "table" in payload or "table_kind" in payload:
                raise InvalidFigureCardError(
                    InvalidFigureCardReason.UNEXPECTED_TABLE,
                    f"figure card of {card.metadata.kind} has no table",
                )
        case unreachable:
            assert_never(unreachable)


def write_figure_card(path: Path, card: FigureCard) -> None:
    """Write ``card`` as TOML; the output is a deterministic function of the card."""
    path.write_bytes(tomli_w.dumps(card_payload(card)).encode("utf-8"))


def read_figure_card(path: Path) -> FigureCard:
    """Read and validate the card stored at ``path``."""
    return decode_figure_card(tomllib.loads(path.read_text(encoding="utf-8")))
