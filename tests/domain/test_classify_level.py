"""Event-based report-level classification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neuralls.domain.benchmarking.selection import ConfiguredReference
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
    UnknownPeakMemory,
)
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    GroupSkippedAllCacheHit,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.labels import FineLabel, GlobalLabel, MediumLabel
from neuralls.domain.solver.models.outcomes import Failed, Solved
from neuralls.domain.solver.reporting.classification import classify_level
from neuralls.domain.solver.reporting.report_levels import FineLevel, GlobalLevel, MediumLevel
from neuralls.shared.types import (
    DatasetId,
    GroupId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)


@pytest.fixture
def method() -> PreconditionerName:
    return PreconditionerName("jacobi")


def _tag(ordinal: int, *, recipe: str = "gauss", matrix: int = 0, dataset: str = "ds") -> SystemTag:
    return SystemTag(
        dataset=DatasetId(dataset),
        recipe=recipe,
        matrix_id=MatrixId(matrix),
        rhs_ordinal=ordinal,
        system_id=SystemId(f"{dataset}-{recipe}-{matrix}-{ordinal}"),
    )


def _setup(method: PreconditionerName, matrix: int = 0) -> MatrixSetupEvent:
    return MatrixSetupEvent(
        event_id=SetupEventId(f"{method}:ds:{matrix}"),
        dataset=DatasetId("ds"),
        matrix_id=MatrixId(matrix),
        method=method,
        setup_s=2.0,
        samples_s=(2.0,),
        peak_memory=UnknownPeakMemory(),
        outcome=SetupCompleted(),
    )


def _attempt(
    tag: SystemTag,
    method: PreconditionerName,
    *,
    converged: bool = True,
) -> SolveAttempt:
    outcome = (
        SolveConverged(iterations=3, final_relative_residual=1e-9)
        if converged
        else SolveFailed(iterations=100, reason="iteration limit")
    )
    return SolveAttempt(
        setup_event_id=SetupEventId(f"{method}:ds:{tag.matrix_id}"),
        system_id=tag.system_id,
        dataset=tag.dataset,
        recipe=tag.recipe,
        matrix_id=tag.matrix_id,
        rhs_ordinal=tag.rhs_ordinal,
        method=method,
        solve_s=0.5,
        samples_s=(0.5,),
        peak_memory=UnknownPeakMemory(),
        outcome=outcome,
    )


def test_counts_planned_attempted_converged_and_matrices_independently(method) -> None:
    tags = (_tag(0, matrix=3), _tag(1, matrix=3), _tag(0, matrix=7))
    setups = (_setup(method, 3), _setup(method, 7))
    attempts = (_attempt(tags[0], method), _attempt(tags[1], method, converged=False))

    (outcome,) = classify_level(tags, setups, attempts, GlobalLevel())

    assert isinstance(outcome, GroupAggregated)
    assert (outcome.n_planned_rhs, outcome.n_matrices) == (3, 2)
    summary = outcome.convergence[method]
    assert (summary.n_planned_rhs, summary.n_attempted_rhs, summary.n_converged) == (3, 2, 1)
    assert outcome.group_id == GroupId("global")
    assert outcome.label == GlobalLabel(("gauss",))


def test_system_cost_rows_become_explicit_solved_and_failed_outcomes(method) -> None:
    tags = (_tag(0), _tag(1))
    setup = _setup(method)
    attempts = (_attempt(tags[0], method), _attempt(tags[1], method, converged=False))

    (group,) = classify_level(tags, (setup,), attempts, FineLevel())

    assert isinstance(group, RecipeProfileReport)
    assert group.outcomes[method] == {
        tags[0].system_id: Solved(
            MatrixId(0), 3, solve_s=0.5, amortized_online_s=1.5, dataset=DatasetId("ds")
        ),
        tags[1].system_id: Failed(MatrixId(0), 100, "iteration limit"),
    }


def test_setup_failure_is_reported_as_zero_attempts_not_zero_planned(method) -> None:
    tags = (_tag(0), _tag(1))
    failed_setup = replace(_setup(method), outcome=SetupFailed(reason="factorization failed"))

    (group,) = classify_level(tags, (failed_setup,), (), FineLevel())

    assert isinstance(group, RecipeProfileReport)
    summary = group.convergence[method]
    assert (summary.n_planned_rhs, summary.n_attempted_rhs, summary.n_converged) == (2, 0, 0)
    assert summary.iterations is None
    assert set(group.outcomes[method]) == {tag.system_id for tag in tags}
    assert all(isinstance(outcome, Failed) for outcome in group.outcomes[method].values())


def test_fine_summary_allocates_setup_using_complete_matrix_membership(method) -> None:
    tags = (_tag(0, recipe="gauss"), _tag(1, recipe="sine"))
    setup = _setup(method)
    attempts = tuple(_attempt(tag, method) for tag in tags)

    reports = classify_level(tags, (setup,), attempts, FineLevel())
    assert all(isinstance(report, RecipeProfileReport) for report in reports)

    fine_reports = [report for report in reports if isinstance(report, RecipeProfileReport)]
    assert [report.summaries[0].setup_total_s for report in fine_reports] == [1.0, 1.0]
    assert sum(report.summaries[0].setup_total_s for report in fine_reports) == pytest.approx(2.0)


def test_setup_failure_creates_asymmetric_censored_outcomes(method) -> None:
    tags = (_tag(0), _tag(1))
    reference = _setup(method)
    failed = replace(_setup(PreconditionerName("amg")), outcome=SetupFailed("failed"))
    attempts = tuple(_attempt(tag, method) for tag in tags)

    (group,) = classify_level(
        tags,
        (reference, failed),
        attempts,
        FineLevel(),
        ConfiguredReference(method),
    )

    assert isinstance(group, RecipeProfileReport)
    comparison = group.comparisons[PreconditionerName("amg")].paired
    assert comparison.n_reference_only == 2
    assert comparison.n_method_only == 0


def test_unmeasured_group_retains_planned_membership(method) -> None:
    tags = (_tag(0), _tag(1))

    (group,) = classify_level(tags, (), (), GlobalLevel())

    assert isinstance(group, GroupSkippedAllCacheHit)
    assert (group.n_planned_rhs, group.n_matrices) == (2, 1)


def test_interrupted_method_is_retained_in_a_group_without_its_setup(method) -> None:
    """The workload plan, rather than observed setup events, defines method population."""
    reference = PreconditionerName("reference")
    candidate = PreconditionerName("candidate")
    tags = (_tag(0, recipe="a", matrix=0), _tag(0, recipe="b", matrix=1))
    setups = (_setup(reference, 0), _setup(reference, 1), _setup(candidate, 0))
    attempts = (
        _attempt(tags[0], reference),
        _attempt(tags[1], reference),
        _attempt(tags[0], candidate),
    )

    reports = classify_level(
        tags,
        setups,
        attempts,
        FineLevel(),
        ConfiguredReference(reference),
        planned_by_method={reference: tags, candidate: tags},
    )

    by_recipe = {
        report.label.recipe: report for report in reports if isinstance(report, RecipeProfileReport)
    }
    interrupted = by_recipe["b"]
    assert interrupted.convergence[candidate].n_attempted_rhs == 0
    assert interrupted.comparisons[candidate].paired.n_reference_only == 1


def test_fine_and_medium_labels_preserve_recipe_membership(method) -> None:
    tags = (_tag(0), _tag(0, recipe="sine"), _tag(0, dataset="other"))
    setups = (
        _setup(method),
        replace(
            _setup(method),
            event_id=SetupEventId(f"{method}:other:0"),
            dataset=DatasetId("other"),
        ),
    )
    attempts = tuple(
        replace(
            _attempt(tag, method),
            setup_event_id=(
                SetupEventId(f"{method}:other:0")
                if tag.dataset == DatasetId("other")
                else SetupEventId(f"{method}:ds:0")
            ),
        )
        for tag in tags
    )

    fine = classify_level(tags, setups, attempts, FineLevel())
    medium = classify_level(tags, setups, attempts, MediumLevel())

    assert [group.label for group in fine] == [
        FineLabel(DatasetId("ds"), "gauss"),
        FineLabel(DatasetId("ds"), "sine"),
        FineLabel(DatasetId("other"), "gauss"),
    ]
    assert [group.label for group in medium] == [
        MediumLabel(DatasetId("ds"), ("gauss", "sine")),
        MediumLabel(DatasetId("other"), ("gauss",)),
    ]
    assert all(isinstance(group, RecipeProfileReport) for group in fine)
    assert [
        group.recipe_membership for group in fine if isinstance(group, RecipeProfileReport)
    ] == [
        ("gauss",),
        ("sine",),
        ("gauss",),
    ]
    assert all(isinstance(group, WorkloadMixtureReport) for group in medium)
    assert [
        group.recipe_membership for group in medium if isinstance(group, WorkloadMixtureReport)
    ] == [("gauss", "sine"), ("gauss",)]

    global_report = classify_level(tags, setups, attempts, GlobalLevel())[0]
    assert isinstance(global_report, WorkloadMixtureReport)
    assert global_report.dataset_membership == (DatasetId("ds"), DatasetId("other"))
    assert global_report.recipe_counts == (2, 1)


def test_removed_cost_models_are_absent_from_event_classification(method) -> None:
    tag = _tag(0)

    (group,) = classify_level((tag,), (_setup(method),), (_attempt(tag, method),), FineLevel())

    assert isinstance(group, RecipeProfileReport)
    summary = group.convergence[method]
    solved = group.outcomes[method][tag.system_id]
    for value in (group, summary, solved):
        assert not hasattr(value, "ert_seconds")
        assert not hasattr(value, "recurring_s")
        assert not hasattr(value, "setup_sharing")


def test_workload_mixture_exposes_no_profile_or_paired_inputs(method) -> None:
    tag = _tag(0)

    (group,) = classify_level((tag,), (_setup(method),), (_attempt(tag, method),), GlobalLevel())

    assert isinstance(group, WorkloadMixtureReport)
    assert not hasattr(group, "outcomes")
    assert not hasattr(group, "comparisons")
