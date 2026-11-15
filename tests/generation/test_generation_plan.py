"""Test count-based generation plan parsing."""

from __future__ import annotations

from typing import Any

import pytest

from neuralls.domain.generation.plan import (
    GenerationPlan,
    StrategySpec,
    canonicalize_strategy_name,
    parse_generation_plan,
)


def _spec(name: str, samples: int) -> StrategySpec:
    return StrategySpec(name, name, samples, {})


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        pytest.param("rhs-archive", "rhs_archive", id="dashes"),
        pytest.param("RHS_ARCHIVE", "rhs_archive", id="case"),
        pytest.param("rhs_bank", "rhs_archive", id="alias-rhs-bank"),
        pytest.param("rhs_repository", "rhs_archive", id="alias-rhs-repository"),
        pytest.param("solution_bank", "solution_archive", id="alias-solution-bank"),
        pytest.param("solution_repository", "solution_archive", id="alias-solution-repository"),
        pytest.param("random", "random", id="standard-random"),
        pytest.param("krylov", "krylov", id="standard-krylov"),
    ],
)
def test_canonicalize_strategy_name(raw: str, canonical: str) -> None:
    assert canonicalize_strategy_name(raw) == canonical


@pytest.mark.parametrize(
    "samples",
    [pytest.param(1000, id="positive"), pytest.param(-1, id="all"), pytest.param(0, id="skip")],
)
def test_strategy_spec_accepts_positive_all_and_skip_counts(samples: int) -> None:
    assert StrategySpec("random", "random", samples, {}).samples == samples


@pytest.mark.parametrize("samples", [-5, -100])
def test_strategy_spec_rejects_negative_counts_other_than_all(samples: int) -> None:
    with pytest.raises(ValueError):
        StrategySpec("random", "random", samples, {})


@pytest.mark.parametrize(
    "strategies",
    [
        pytest.param({}, id="empty"),
        pytest.param({"random": _spec("random", 0), "krylov": _spec("krylov", 0)}, id="all-zero"),
    ],
)
def test_plan_rejects_no_contributing_strategy(strategies: dict[str, StrategySpec]) -> None:
    with pytest.raises(ValueError):
        GenerationPlan(strategies=strategies)


@pytest.mark.parametrize(
    ("strategies", "rhs_archive", "solution_archive", "synthetic"),
    [
        pytest.param([_spec("random", 100)], None, None, {"random"}, id="random-only"),
        pytest.param(
            [_spec("rhs_archive", 100)], _spec("rhs_archive", 100), None, set(), id="rhs-archive"
        ),
        pytest.param(
            [_spec("solution_archive", -1)],
            None,
            _spec("solution_archive", -1),
            set(),
            id="solution-archive-all",
        ),
        pytest.param(
            [_spec("random", 500), _spec("krylov", 300), _spec("rhs_archive", 200)],
            _spec("rhs_archive", 200),
            None,
            {"random", "krylov"},
            id="synthetic-excludes-rhs-archive",
        ),
        pytest.param(
            [_spec("random", 100), _spec("rhs_archive", 50), _spec("solution_archive", 50)],
            _spec("rhs_archive", 50),
            _spec("solution_archive", 50),
            {"random"},
            id="synthetic-excludes-both-archives",
        ),
    ],
)
def test_plan_separates_archive_and_synthetic_strategies(
    strategies: list[StrategySpec],
    rhs_archive: StrategySpec | None,
    solution_archive: StrategySpec | None,
    synthetic: set[str],
) -> None:
    plan = GenerationPlan(strategies={spec.raw_name: spec for spec in strategies})

    assert plan.rhs_archive == rhs_archive
    assert plan.solution_archive == solution_archive
    assert set(plan.synthetic) == synthetic


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="missing-strategy-array"),
        pytest.param({"strategy": []}, id="empty-strategy-array"),
        pytest.param({"strategy": "not_an_array"}, id="strategy-not-array"),
        pytest.param({"strategy": ["not_a_table"]}, id="entry-not-table"),
        pytest.param({"strategy": [{"samples": 100}]}, id="missing-name"),
        pytest.param({"strategy": [{"name": "", "samples": 100}]}, id="empty-name"),
        pytest.param({"strategy": [{"name": "random"}]}, id="missing-samples"),
        pytest.param({"strategy": [{"name": "random", "samples": "many"}]}, id="non-integer"),
        pytest.param({"strategy": [{"name": "random", "samples": -5}]}, id="negative-samples"),
    ],
)
def test_parse_rejects_invalid_config(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_generation_plan(config)


@pytest.mark.parametrize(
    ("entries", "samples", "synthetic"),
    [
        pytest.param(
            [{"name": "random", "samples": 100.0}], {"random": 100}, {"random"}, id="float-samples"
        ),
        pytest.param(
            [{"name": "random", "samples": 500}, {"name": "krylov", "samples": 300}],
            {"random": 500, "krylov": 300},
            {"random", "krylov"},
            id="several-strategies",
        ),
        pytest.param(
            [{"name": "random", "samples": 500}, {"name": "random", "samples": 300}],
            {"random": 800},
            {"random"},
            id="duplicates-sum",
        ),
        pytest.param(
            [
                {"name": "solution_archive", "samples": 500, "solutions_glob": "a/*.txt"},
                {"name": "solution_archive", "samples": -1, "solutions_glob": "b/*.txt"},
            ],
            {"solution_archive": -1},
            set(),
            id="duplicate-all-last",
        ),
        pytest.param(
            [
                {"name": "solution_archive", "samples": -1, "solutions_glob": "a/*.txt"},
                {"name": "solution_archive", "samples": 500, "solutions_glob": "b/*.txt"},
            ],
            {"solution_archive": -1},
            set(),
            id="duplicate-all-first",
        ),
        pytest.param(
            [{"name": "random", "samples": 0}, {"name": "random", "samples": 300}],
            {"random": 300},
            {"random"},
            id="duplicate-zero-uses-the-other",
        ),
        pytest.param(
            [
                {"name": "random", "samples": 0},
                {"name": "random", "samples": 0},
                {"name": "krylov", "samples": 100},
            ],
            {"random": 0, "krylov": 100},
            {"random", "krylov"},
            id="duplicate-zero-stays-zero",
        ),
        pytest.param(
            [{"name": "rhs-bank", "samples": 100}],
            {"rhs_archive": 100},
            set(),
            id="legacy-alias",
        ),
        pytest.param(
            [
                {"name": "rhs_archive", "samples": 100},
                {"name": "solution_archive", "samples": -1},
                {"name": "random", "samples": 500},
            ],
            {"rhs_archive": 100, "solution_archive": -1, "random": 500},
            {"random"},
            id="archives-are-not-synthetic",
        ),
    ],
)
def test_parse_counts_per_strategy(
    entries: list[dict[str, Any]], samples: dict[str, int], synthetic: set[str]
) -> None:
    plan = parse_generation_plan({"strategy": entries})

    assert {name: spec.samples for name, spec in plan.strategies.items()} == samples
    assert set(plan.synthetic) == synthetic


def test_parse_alias_resolves_to_the_canonical_archive() -> None:
    plan = parse_generation_plan({"strategy": [{"name": "rhs-bank", "samples": 100}]})

    assert plan.rhs_archive == plan.strategies["rhs_archive"]


@pytest.mark.parametrize(
    ("entries", "name", "options"),
    [
        pytest.param(
            [
                {
                    "name": "residuals",
                    "samples": 10000,
                    "cg_iters": 50,
                    "solutions_glob": "data/path/*.txt",
                }
            ],
            "residuals",
            {"cg_iters": 50, "solutions_glob": "data/path/*.txt"},
            id="name-and-samples-are-not-options",
        ),
        pytest.param(
            [
                {"name": "random", "samples": 500, "scale": 1.0, "seed": 42},
                {"name": "random", "samples": 300, "scale": 2.0, "offset": 10},
            ],
            "random",
            {"scale": 2.0, "seed": 42, "offset": 10},
            id="duplicates-merge-with-later-value-winning",
        ),
    ],
)
def test_parse_extracts_options(
    entries: list[dict[str, Any]], name: str, options: dict[str, Any]
) -> None:
    plan = parse_generation_plan({"strategy": entries})

    assert plan.strategies[name].options == options
