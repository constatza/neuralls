"""Response series: how a method family's metric moves along the one fact it sweeps.

A *family* is the methods of one group that share a base name (POD-CG, POD-JP). Its *swept
fact* is the one fact whose value differs between them, read from structured facts, never
from rendered text. Method facts and dataset-strategy facts are kept on separate sides, so a
key both carry (for example a count) never merges. A family is a response series only when
it has at least ``MIN_RESPONSE_POINTS`` methods fit on data of one context and its swept fact
is numeric on every method; anything else (a categorical axis, two facts changing at once,
data from different sources) is left to the per-method figures.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import assert_never

import polars as pl

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
    StandaloneMethodPresentation,
    description_facts,
)
from neuralls.domain.presentation.provenance import DatasetSourceProvenance
from neuralls.domain.solver.reporting.cohorts import cohort_schema
from neuralls.shared.scientific_description.catalog import FactRole, fact_definition
from neuralls.shared.scientific_description.models import (
    DecimalFact,
    FieldKey,
    IntegerFact,
    ScientificFact,
)
from neuralls.shared.system_scaling import SystemScaling
from neuralls.shared.types import GenerationStrategyKind, PreconditionerName

MIN_RESPONSE_POINTS = 3
"""Fewest methods a family needs before its swept fact is drawn as a response curve."""

LOG_X_SPAN = 10.0
"""Ratio of the largest to the smallest swept value from which a response axis is logarithmic."""

RESPONSE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "family": pl.String,
    "side": pl.String,
    "fact": pl.String,
    "x": pl.Float64,
    "method": pl.String,
    "metric": pl.String,
    "n_planned": pl.Int64,
    "n_defined": pl.Int64,
    "n_unconverged": pl.Int64,
    "n_solver_error": pl.Int64,
    "n_setup_failed": pl.Int64,
    "n_not_attempted": pl.Int64,
    "q25": pl.Float64,
    "q50": pl.Float64,
    "q75": pl.Float64,
}
"""One row per method of a response series, sorted by family and x. The quantiles are null
for a method with no defined system, which keeps its row and its outcome counts."""


COHORT_RESPONSE_SCHEMA = cohort_schema(RESPONSE_SCHEMA)
"""The response table of a report level: one ``RESPONSE_SCHEMA`` block per cohort."""


class FactSide(StrEnum):
    """Whether a swept fact belongs to the method or to the strategy that made its data."""

    METHOD = "method"
    DATASET = "dataset"


@dataclass(frozen=True, slots=True)
class SweptFact:
    """The fact a family varies, on its side."""

    side: FactSide
    key: FieldKey


@dataclass(frozen=True, slots=True)
class ResponsePoint:
    """Where one method sits on its family's response curve."""

    family: str
    fact: SweptFact
    x: float


@dataclass(frozen=True, slots=True)
class _DataContext:
    """What must be equal for two methods' data to differ only in a strategy fact."""

    source: DatasetSourceProvenance
    scaling: SystemScaling
    root_seed: int
    strategy: GenerationStrategyKind


@dataclass(frozen=True, slots=True)
class _Profile:
    family: str
    context: _DataContext
    facts: Mapping[SweptFact, ScientificFact]


NO_RESPONSE_POINTS: Mapping[PreconditionerName, ResponsePoint] = MappingProxyType({})
"""A group with no response series."""


def response_points(
    methods: Mapping[PreconditionerName, ConfiguredMethodPresentation],
) -> dict[PreconditionerName, ResponsePoint]:
    """The response point of every method that belongs to a numeric response series.

    Methods with no profile (standalone, or fit on a mixture of strategies) and families that
    are not a response series have no point.
    """
    families: defaultdict[str, dict[PreconditionerName, _Profile]] = defaultdict(dict)
    for name, configured in methods.items():
        profile = _profile(configured)
        if profile is not None:
            families[profile.family][name] = profile
    return {
        name: point
        for family, members in families.items()
        for name, point in _family_points(family, members).items()
    }


def response_table(
    summary: pl.DataFrame, points: Mapping[PreconditionerName, ResponsePoint]
) -> pl.DataFrame:
    """Join each point with its method's summary row (``SUMMARY_SCHEMA``), with ``RESPONSE_SCHEMA``.

    Only methods with a point and a summary row are kept.
    """
    located = pl.DataFrame(
        [
            {
                "family": point.family,
                "side": str(point.fact.side),
                "fact": point.fact.key.value,
                "x": point.x,
                "method": str(name),
            }
            for name, point in points.items()
        ],
        schema={
            column: RESPONSE_SCHEMA[column] for column in ("family", "side", "fact", "x", "method")
        },
    )
    return (
        located.join(summary, on="method", how="inner")
        .select(list(RESPONSE_SCHEMA))
        .sort("family", "x")
    )


def _profile(configured: ConfiguredMethodPresentation) -> _Profile | None:
    """A method's comparable facts, or ``None`` when its data has no single strategy."""
    match configured:
        case StandaloneMethodPresentation():
            return None
        case DatasetQualifiedMethodPresentation(method=method, dataset=dataset):
            match dataset.generation.strategies:
                case (strategy,):
                    generation = dataset.generation
                    return _Profile(
                        family=method.base_name,
                        context=_DataContext(
                            dataset.source, generation.scaling, generation.root_seed, strategy.kind
                        ),
                        facts={
                            **_sided(FactSide.METHOD, description_facts(method)),
                            **_sided(FactSide.DATASET, strategy.facts),
                        },
                    )
                case _:
                    return None
        case _:
            assert_never(configured)


def _sided(side: FactSide, facts: tuple[ScientificFact, ...]) -> dict[SweptFact, ScientificFact]:
    # A seed is how a draw was made, not a parameter of the method, so it is never an axis.
    return {
        SweptFact(side, fact.key): fact
        for fact in facts
        if fact_definition(fact.key).role is not FactRole.REPRODUCIBILITY
    }


def _family_points(
    family: str, members: Mapping[PreconditionerName, _Profile]
) -> dict[PreconditionerName, ResponsePoint]:
    if len(members) < MIN_RESPONSE_POINTS:
        return {}
    profiles = list(members.values())
    if any(profile.context != profiles[0].context for profile in profiles):
        return {}
    keys = {key for profile in profiles for key in profile.facts}
    swept = [key for key in keys if len({profile.facts.get(key) for profile in profiles}) > 1]
    match swept:
        case [fact]:
            pass
        case _:
            return {}
    xs: dict[PreconditionerName, float] = {}
    for name, profile in members.items():
        match profile.facts.get(fact):
            case IntegerFact(value=value) | DecimalFact(value=value):
                xs[name] = float(value)
            case _:
                return {}
    return {name: ResponsePoint(family, fact, x) for name, x in xs.items()}
