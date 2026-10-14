"""Tests for ``domain.solver.comparison.classify_level``.

Pure classification: given every planned system's tag and the results of those
that produced some, decide per report group which of the two
``ComparisonGroupOutcome`` cases applies. No MLflow or I/O involved.
"""

from __future__ import annotations

import numpy as np

from neuralls.domain.solver.comparison import classify_level
from neuralls.domain.solver.metrics_table import FineLevel, GlobalLevel, MediumLevel, SystemTag
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    FineLabel,
    GlobalLabel,
    GroupAggregated,
    GroupSkippedAllCacheHit,
    MediumLabel,
    SetupSharingPolicy,
)
from neuralls.shared.types import DatasetId, GroupId, MatrixId


def _result(name: str) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 1e-9],
        residual_history_abs=[1.0, 1e-9],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=1.0,
        solve_time_seconds=0.5,
    )


def _tag(ordinal: int, *, recipe: str = "gauss", matrix: int = 0, dataset: str = "ds") -> SystemTag:
    return SystemTag(
        dataset=DatasetId(dataset),
        recipe=recipe,
        matrix_id=MatrixId(matrix),
        rhs_ordinal=ordinal,
        system_id=f"{dataset}-{recipe}-{matrix}-{ordinal}",
    )


def _measured(tags: list[SystemTag]) -> dict[SystemTag, dict[str, CGComparisonResult]]:
    return {tag: {"jacobi": _result("jacobi")} for tag in tags}


_SHARING = {"jacobi": SetupSharingPolicy(share_setup=False, share_generation=False)}


def test_all_systems_contributing_classifies_as_fully_aggregated() -> None:
    tags = [_tag(0), _tag(1)]

    (outcome,) = classify_level(tags, _measured(tags), _SHARING, GlobalLevel())

    assert isinstance(outcome, GroupAggregated)
    assert outcome.n_aggregated == 2
    assert outcome.n_in_group == 2
    assert outcome.n_matrices == 1
    assert set(outcome.projected) == {"jacobi"}
    assert outcome.label == GlobalLabel(mixture_of=("gauss",))
    assert outcome.group_id == GroupId("global")
    assert outcome.time_to_solution_by_system == {
        "jacobi": {tags[0].system_id: 1.5, tags[1].system_id: 1.5}
    }


def test_some_systems_cache_hits_classifies_as_partially_aggregated() -> None:
    """Fewer measured systems than planned — still GroupAggregated, with the gap recorded."""
    tags = [_tag(0), _tag(1), _tag(2)]

    (outcome,) = classify_level(tags, _measured(tags[:1]), _SHARING, GlobalLevel())

    assert isinstance(outcome, GroupAggregated)
    assert outcome.n_aggregated == 1
    assert outcome.n_in_group == 3
    assert outcome.projected["jacobi"].solve_time_std_seconds is None


def test_every_system_a_cache_hit_classifies_as_skipped() -> None:
    tags = [_tag(i) for i in range(4)]

    (outcome,) = classify_level(tags, {}, _SHARING, GlobalLevel())

    assert isinstance(outcome, GroupSkippedAllCacheHit)
    assert outcome.group_id == GroupId("global")
    assert outcome.label == GlobalLabel(mixture_of=("gauss",))
    assert outcome.n_in_group == 4


def test_fine_level_yields_one_outcome_per_dataset_and_recipe() -> None:
    tags = [_tag(0), _tag(1), _tag(0, recipe="sine"), _tag(0, dataset="other")]

    outcomes = classify_level(tags, _measured(tags), _SHARING, FineLevel())

    assert [outcome.label for outcome in outcomes] == [
        FineLabel(DatasetId("ds"), "gauss"),
        FineLabel(DatasetId("ds"), "sine"),
        FineLabel(DatasetId("other"), "gauss"),
    ]
    assert [outcome.n_in_group for outcome in outcomes] == [2, 1, 1]


def test_medium_level_pools_a_datasets_recipes_and_names_the_mixture() -> None:
    tags = [_tag(0), _tag(0, recipe="sine"), _tag(0, dataset="other")]

    outcomes = classify_level(tags, _measured(tags), _SHARING, MediumLevel())

    assert [outcome.label for outcome in outcomes] == [
        MediumLabel(DatasetId("ds"), ("gauss", "sine")),
        MediumLabel(DatasetId("other"), ("gauss",)),
    ]


def test_a_fully_cached_group_is_skipped_while_its_siblings_are_aggregated() -> None:
    tags = [_tag(0), _tag(0, recipe="sine")]

    outcomes = classify_level(tags, _measured(tags[:1]), _SHARING, FineLevel())

    assert [type(outcome) for outcome in outcomes] == [GroupAggregated, GroupSkippedAllCacheHit]


def test_systems_count_distinct_matrices_not_rhs_draws() -> None:
    tags = [_tag(0, matrix=3), _tag(1, matrix=3), _tag(0, matrix=7), _tag(1, matrix=7)]

    (outcome,) = classify_level(tags, _measured(tags), _SHARING, MediumLevel())

    assert isinstance(outcome, GroupAggregated)
    assert (outcome.n_aggregated, outcome.n_matrices) == (4, 2)


def test_group_ids_are_unique_across_levels() -> None:
    tags = [_tag(0)]
    ids = [
        outcome.group_id
        for level in (FineLevel(), MediumLevel(), GlobalLevel())
        for outcome in classify_level(tags, _measured(tags), _SHARING, level)
    ]

    assert len(set(ids)) == 3
