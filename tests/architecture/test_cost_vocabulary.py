"""Every cost name is composed of the keywords of the cost vocabulary, so a new name cannot drift.

The vocabulary table in ``domain/solver/README.md`` is the single source; this test reads its
keywords from there and checks the names code stores: the timed metrics, the workload-summary
fields and the stems of the figures drawing a timed metric.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
)
from neuralls.domain.solver.reporting.metrics import METRICS, Unit
from neuralls.platform.reporting.figure_metadata import WORKLOAD_SCHEMA
from neuralls.platform.reporting.figure_recipes import group_recipes

VOCABULARY_DOC = Path(__file__).parents[2] / "src" / "neuralls" / "domain" / "solver" / "README.md"
_SECTION = "### Cost vocabulary"
_ROW = re.compile(r"^\| `(?P<keyword>[a-z_<>]+)` \| (?P<kind>[a-z ]+) \|")
_STAGE_KIND = "stage"
_SECONDS_SUFFIX = "_s"
_FIGURE_FORM_WORDS = frozenset(
    {
        "bars",
        "breakdown",
        "cdf",
        "common",
        "cost",
        "distribution",
        "ecdf",
        "means",
        "own",
        "paired",
        "performance",
        "profile",
        "ratio",
        "scatter",
        "speed",
    }
)
"""Words naming what a figure draws or its population, not the quantity drawn."""
_KIND_PREFIXES = ("time_means_", "time_distribution_")
"""Kinds whose stem words name the stage drawn, so their leading ``time`` names no quantity."""


def _vocabulary() -> dict[str, str]:
    """Each keyword of the README table, stripped of its ``_`` and ``<stage>`` affixes, by kind."""
    text = VOCABULARY_DOC.read_text(encoding="utf-8")
    section = text.split(_SECTION, 1)[1].split("\n#", 1)[0]
    rows = (_ROW.match(line) for line in section.splitlines())
    return {
        row["keyword"].removesuffix("_<stage>").strip("_"): row["kind"]
        for row in rows
        if row is not None
    }


VOCABULARY = _vocabulary()
STAGES = frozenset(keyword for keyword, kind in VOCABULARY.items() if kind == _STAGE_KIND)

_TIMED_METRICS = [spec for spec in METRICS.values() if spec.unit is Unit.SECONDS]
COST_NAMES = sorted(
    {
        *(spec.column for spec in _TIMED_METRICS),
        *(
            field.name
            for summary in (CompleteWorkloadSummary, IncompleteWorkloadSummary)
            for field in fields(summary)
            if field.name.endswith(_SECONDS_SUFFIX)
        ),
        *(column for column in WORKLOAD_SCHEMA if column.endswith(_SECONDS_SUFFIX)),
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


def _quantity_words(stem: str) -> list[str]:
    for prefix in _KIND_PREFIXES:
        stem = stem.removeprefix(prefix)
    return [word for word in stem.split("_") if word not in _FIGURE_FORM_WORDS]


@pytest.mark.parametrize(
    "words",
    [
        *(
            pytest.param(name.removesuffix(_SECONDS_SUFFIX).split("_"), id=name)
            for name in COST_NAMES
        ),
        *(pytest.param(_quantity_words(stem), id=stem) for stem in COST_STEMS),
    ],
)
def test_a_cost_name_is_composed_of_vocabulary_keywords_around_one_stage(
    words: list[str],
) -> None:
    assert (set(words) - set(VOCABULARY), len(STAGES.intersection(words)) >= 1) == (set(), True)
