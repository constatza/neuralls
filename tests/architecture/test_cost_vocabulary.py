"""Every cost name follows the cost-vocabulary grammar, so a new name cannot drift.

The vocabulary table in ``domain/solver/README.md`` is the single source; this test builds the
name grammar from its rows and checks the names code stores: the timed metrics, the
workload-summary fields, the seconds and break-even columns of the stored tables, the MLflow cost
keys and the stems of the figures drawing a timed metric.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from neuralls.domain.benchmarking.cost_curves import COST_CURVE_SCHEMA
from neuralls.domain.benchmarking.paired import BREAK_EVEN_SCHEMA
from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
)
from neuralls.domain.solver.reporting.metrics import METRICS, Unit
from neuralls.platform.reporting.figure_metadata import (
    FULL_COST_BREAKDOWN_SCHEMA,
    WORKLOAD_SCHEMA,
)
from neuralls.platform.reporting.figure_recipes import group_recipes
from neuralls.platform.tracking.method_sink import OFFLINE_COST_KEYS, WORKLOAD_COST_KEYS

VOCABULARY_DOC = Path(__file__).parents[2] / "src" / "neuralls" / "domain" / "solver" / "README.md"
_SECTION = "### Cost vocabulary"
_ROW = re.compile(r"^\| `(?P<keyword>[a-z_<>]+)` \| (?P<kind>[a-z]+) \|")
_SECONDS_SUFFIX = "_s"
_BREAKEVEN_PREFIX = "breakeven_"

_FIGURE_FORM_PREFIXES = (
    "ecdf_",
    "paired_",
    "performance_profile_",
    "speed_ratio_",
    "time_distribution_",
    "time_means_",
)
"""Stem prefixes naming the figure form; a stem has at most one."""
_FIGURE_FORM_SUFFIXES = (
    "_bars",
    "_cdf",
    "_common",
    "_cost_breakdown",
    "_means",
    "_own",
    "_scatter",
)
"""Stem suffixes naming the figure form or its population; a stem has at most one."""


def _keywords_by_kind() -> dict[str, list[str]]:
    """The README table's keywords, stripped of their ``_`` and ``<stage>`` affixes, by kind."""
    text = VOCABULARY_DOC.read_text(encoding="utf-8")
    section = text.split(_SECTION, 1)[1].split("\n#", 1)[0]
    by_kind: dict[str, list[str]] = {}
    for row in (_ROW.match(line) for line in section.splitlines()):
        if row is not None:
            keyword = row["keyword"].removesuffix("_<stage>").strip("_")
            by_kind.setdefault(row["kind"], []).append(keyword)
    return by_kind


def _grammar() -> re.Pattern[str]:
    """``[breakeven_][scope_][filter_]stage[_amortized][_statistic]``, without the unit suffix."""
    kinds = _keywords_by_kind()

    def one_of(kind: str) -> str:
        return "|".join(kinds[kind])

    return re.compile(
        rf"(?:(?:{one_of('derived')})_)?(?:(?:{one_of('scope')})_)?(?:(?:{one_of('filter')})_)?"
        rf"(?:{one_of('stage')})(?:_(?:{one_of('modifier')}))?(?:_(?:{one_of('statistic')}))?"
    )


GRAMMAR = _grammar()

_STORED_COLUMNS = (
    *WORKLOAD_SCHEMA,
    *BREAK_EVEN_SCHEMA,
    *COST_CURVE_SCHEMA,
    *FULL_COST_BREAKDOWN_SCHEMA,
)
COST_NAMES = sorted(
    {
        *(spec.column for spec in METRICS.values() if spec.unit is Unit.SECONDS),
        *(
            field.name
            for summary in (CompleteWorkloadSummary, IncompleteWorkloadSummary)
            for field in fields(summary)
            if field.name.endswith(_SECONDS_SUFFIX)
        ),
        *(
            column
            for column in _STORED_COLUMNS
            if column.endswith(_SECONDS_SUFFIX) or column.startswith(_BREAKEVEN_PREFIX)
        ),
        *WORKLOAD_COST_KEYS,
        *OFFLINE_COST_KEYS,
    }
)
COST_STEMS = sorted(
    {
        variant.stem
        for recipe in group_recipes()
        for variant in recipe.variants
        if METRICS[variant.metric].unit is Unit.SECONDS
    }
)


def _quantity(name: str) -> str:
    """A cost name without its unit: a ``breakeven_`` count carries none."""
    if name.startswith(_BREAKEVEN_PREFIX) and not name.endswith(_SECONDS_SUFFIX):
        return name
    assert name.endswith(_SECONDS_SUFFIX), name
    return name.removesuffix(_SECONDS_SUFFIX)


def _stem_quantity(stem: str) -> str:
    """A figure stem without its figure-form prefix and suffix."""
    prefix = next((p for p in _FIGURE_FORM_PREFIXES if stem.startswith(p)), "")
    stem = stem.removeprefix(prefix)
    suffix = next((s for s in _FIGURE_FORM_SUFFIXES if stem.endswith(s)), "")
    return stem.removesuffix(suffix)


@pytest.mark.parametrize(
    "quantity",
    [
        *(pytest.param(_quantity(name), id=name) for name in COST_NAMES),
        *(pytest.param(_stem_quantity(stem), id=stem) for stem in COST_STEMS),
    ],
)
def test_a_cost_name_follows_the_vocabulary_grammar(quantity: str) -> None:
    assert GRAMMAR.fullmatch(quantity) is not None, quantity
