"""Response series: the one numeric fact a method family sweeps, joined with its summary."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from neuralls.domain.benchmarking.response import (
    RESPONSE_SCHEMA,
    FactSide,
    ResponsePoint,
    SweptFact,
    response_points,
    response_table,
)
from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
    StandaloneMethodPresentation,
)
from neuralls.domain.presentation.provenance import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    DerivedSeed,
    StrategyProvenance,
)
from neuralls.domain.solver.reporting.summary import SUMMARY_SCHEMA
from neuralls.shared.scientific_description.catalog import (
    ITERATION_COUNT,
    PROBE_DISTRIBUTION,
    RANK,
    STEPS,
    make_fact,
)
from neuralls.shared.scientific_description.models import ScientificDescription, ScientificFact
from neuralls.shared.system_scaling import NoScaling
from neuralls.shared.types import GenerationStrategyKind, PreconditionerName

CG = GenerationStrategyKind.CG_ERRORS
JP = GenerationStrategyKind.JACOBI_PROBES
type Strategy = tuple[GenerationStrategyKind, tuple[ScientificFact, ...]]
type MethodOn = Callable[..., DatasetQualifiedMethodPresentation]


def _strategy(
    kind: GenerationStrategyKind, parameters: tuple[ScientificFact, ...], rows: int
) -> StrategyProvenance:
    return StrategyProvenance(kind, rows=rows, parameters=parameters, seed=DerivedSeed())


@pytest.fixture
def method_on() -> MethodOn:
    """Build a POD method with rank ``c`` fit on data of the given strategies."""

    def build(
        name: str, c: int, *strategies: Strategy, source: str = "RHC", rows: int = 5000
    ) -> DatasetQualifiedMethodPresentation:
        built = tuple(_strategy(kind, parameters, rows) for kind, parameters in strategies)
        dataset = DatasetPresentation(
            source=DatasetSourceProvenance(ScientificDescription(source)),
            generation=DatasetGenerationProvenance(
                strategies=built,
                generated_rows=rows * len(built),
                scaling=NoScaling(),
                root_seed=42,
            ),
        )
        return DatasetQualifiedMethodPresentation(
            ScientificDescription(name, facts=(make_fact(RANK, c),)), dataset
        )

    return build


def _cg(k: int) -> Strategy:
    return CG, (make_fact(ITERATION_COUNT, k),)


def _jp(p: int, *extra: ScientificFact) -> Strategy:
    return JP, (make_fact(STEPS, p), *extra)


def _names(
    *methods: ConfiguredMethodPresentation,
) -> dict[PreconditionerName, ConfiguredMethodPresentation]:
    return {PreconditionerName(f"m{index}"): method for index, method in enumerate(methods)}


def test_each_family_is_located_on_the_one_fact_it_sweeps(method_on: MethodOn) -> None:
    """A method-side rank sweep and a dataset-side sweep are found per family."""
    methods = {
        **{PreconditionerName(f"cg{c}"): method_on("POD-CG", c, _cg(20)) for c in (10, 50, 100)},
        **{PreconditionerName(f"jp{p}"): method_on("POD-JP", 100, _jp(p)) for p in (1, 5, 10)},
    }

    rank = SweptFact(FactSide.METHOD, RANK)
    sweeps = SweptFact(FactSide.DATASET, STEPS)
    assert response_points(methods) == {
        **{
            PreconditionerName(f"cg{c}"): ResponsePoint("POD-CG", rank, float(c))
            for c in (10, 50, 100)
        },
        **{
            PreconditionerName(f"jp{p}"): ResponsePoint("POD-JP", sweeps, float(p))
            for p in (1, 5, 10)
        },
    }


@pytest.mark.parametrize(
    "family",
    [
        lambda on: (on("POD-CG", 10, _cg(20)), on("POD-CG", 50, _cg(20))),
        lambda on: tuple(on("POD-CG", c, _cg(c)) for c in (10, 20, 50)),
        lambda on: (
            on("POD-JP", 10, _jp(10)),
            on("POD-JP", 10, _jp(10, make_fact(PROBE_DISTRIBUTION, "gaussian"))),
            on("POD-JP", 10, _jp(10, make_fact(PROBE_DISTRIBUTION, "rademacher"))),
        ),
        lambda on: (
            on("POD-CG", 10, _cg(20)),
            on("POD-CG", 50, _cg(20)),
            on("POD-CG", 100, _cg(20), source="Other"),
        ),
        lambda on: tuple(
            StandaloneMethodPresentation(on("POD", c, _cg(20)).method) for c in (10, 50, 100)
        ),
        lambda on: tuple(on("POD", c, _cg(20), _jp(10)) for c in (10, 50, 100)),
    ],
    ids=[
        "two-methods",
        "two-facts-differ",
        "categorical",
        "other-source",
        "standalone",
        "mixture",
    ],
)
def test_a_family_that_is_not_a_numeric_series_has_no_points(
    method_on: MethodOn, family: Callable[[MethodOn], tuple[ConfiguredMethodPresentation, ...]]
) -> None:
    assert response_points(_names(*family(method_on))) == {}


def test_the_table_joins_each_point_with_its_summary_and_keeps_failed_methods() -> None:
    rank = SweptFact(FactSide.METHOD, RANK)
    points = {
        PreconditionerName("c100"): ResponsePoint("POD-CG", rank, 100.0),
        PreconditionerName("c10"): ResponsePoint("POD-CG", rank, 10.0),
    }
    counts = {"metric": "iterations", "population": "own", "n_planned": 4, "n_matrices": 1}
    summary = pl.DataFrame(
        [
            {**counts, "method": "c100", "n_defined": 4, "q25": 3.0, "q50": 4.0, "q75": 5.0},
            {**counts, "method": "c10", "n_defined": 0, "n_unconverged": 4},
            {**counts, "method": "jacobi", "n_defined": 4, "q50": 90.0},
        ],
        schema=SUMMARY_SCHEMA,
    )

    expected = pl.DataFrame(
        [
            {
                "family": "POD-CG",
                "side": "method",
                "fact": "rank",
                "x": 10.0,
                "method": "c10",
                "metric": "iterations",
                "n_planned": 4,
                "n_defined": 0,
                "n_unconverged": 4,
            },
            {
                "family": "POD-CG",
                "side": "method",
                "fact": "rank",
                "x": 100.0,
                "method": "c100",
                "metric": "iterations",
                "n_planned": 4,
                "n_defined": 4,
                "q25": 3.0,
                "q50": 4.0,
                "q75": 5.0,
            },
        ],
        schema=RESPONSE_SCHEMA,
    )
    assert_frame_equal(response_table(summary, points), expected)
