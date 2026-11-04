"""Event-based report-level classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from neuralls.domain.benchmarking.paired import REFERENCE_METRIC, speed_ratio_summary
from neuralls.domain.benchmarking.selection import ConfiguredReference, ReferenceSelection
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredResidual,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
    UnknownPeakMemory,
)
from neuralls.domain.solver.models.group_outcomes import (
    ComparisonGroupOutcome,
    GroupAggregated,
    GroupSkippedAllCacheHit,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.domain.solver.reporting.classification import classify_level
from neuralls.domain.solver.reporting.report_levels import (
    AllMatrixSetsLevel,
    MatrixSetLevel,
    MatrixSetRhsTypeLevel,
    ReportLevel,
)
from neuralls.shared.device import SingleRun
from neuralls.shared.rhs_recipe import RhsRecipe, recipe_key
from neuralls.shared.types import (
    DatasetId,
    GroupId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)
from tests.measurement_support import NO_REFERENCE
from tests.observation_support import observations_of_events
from tests.recipe_support import ALPHA, BETA, GAUSSIAN, SINE


@pytest.fixture
def method() -> PreconditionerName:
    return PreconditionerName("jacobi")


def _tag(
    ordinal: int, *, recipe: RhsRecipe = GAUSSIAN, matrix: int = 0, dataset: str = "ds"
) -> SystemTag:
    return SystemTag(
        dataset=DatasetId(dataset),
        recipe=recipe,
        matrix_id=MatrixId(matrix),
        rhs_ordinal=ordinal,
        system_id=SystemId(f"{dataset}-{recipe_key(recipe)}-{matrix}-{ordinal}"),
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
        residual=MeasuredResidual(rhs_norm=2.0, true_residual_rel=1e-3),
        exact_reference=NO_REFERENCE,
        repeats=SingleRun(),
    )


def _classify(
    tags: Sequence[SystemTag],
    setups: Sequence[MatrixSetupEvent],
    attempts: Sequence[SolveAttempt],
    level: ReportLevel,
    selection: ReferenceSelection | None = None,
    *,
    planned_by_method: Mapping[PreconditionerName, Sequence[SystemTag]] | None = None,
) -> tuple[ComparisonGroupOutcome, ...]:
    """``classify_level`` given the observation view of the same hand-built events."""
    return classify_level(
        tags,
        setups,
        attempts,
        level,
        selection,
        observations=observations_of_events(
            tags, setups, attempts, planned_by_method=planned_by_method
        ),
        planned_by_method=planned_by_method,
    )


def test_counts_planned_attempted_converged_and_matrices_independently(method) -> None:
    tags = (_tag(0, matrix=3), _tag(1, matrix=3), _tag(0, matrix=7))
    setups = (_setup(method, 3), _setup(method, 7))
    attempts = (_attempt(tags[0], method), _attempt(tags[1], method, converged=False))

    (outcome,) = _classify(tags, setups, attempts, AllMatrixSetsLevel())

    assert isinstance(outcome, GroupAggregated)
    assert (outcome.n_planned_rhs, outcome.n_matrices) == (3, 2)
    summary = outcome.convergence[method]
    assert (summary.n_planned_rhs, summary.n_attempted_rhs, summary.n_converged) == (3, 2, 1)
    assert outcome.group_id == GroupId("all")
    assert outcome.label == AllMatrixSetsLabel((GAUSSIAN,))


def test_group_observations_are_the_systems_of_the_group_with_their_outcome(method) -> None:
    tags = (_tag(0), _tag(1))
    setup = _setup(method)
    attempts = (_attempt(tags[0], method), _attempt(tags[1], method, converged=False))

    (group,) = _classify(tags, (setup,), attempts, MatrixSetRhsTypeLevel())

    assert isinstance(group, RecipeProfileReport)
    # setup 2.0 over the 2 planned RHS adds 1.0 to a solve time of 0.5
    assert group.observations.select(
        "system_id", "outcome", "iterations", "solve_s", "online_s"
    ).rows() == [
        (tags[0].system_id, "converged", 3, 0.5, 1.5),
        (tags[1].system_id, "unconverged", 100, 0.5, 1.5),
    ]


def test_setup_failure_is_reported_as_zero_attempts_not_zero_planned(method) -> None:
    tags = (_tag(0), _tag(1))
    failed_setup = replace(_setup(method), outcome=SetupFailed(reason="factorization failed"))

    (group,) = _classify(tags, (failed_setup,), (), MatrixSetRhsTypeLevel())

    assert isinstance(group, RecipeProfileReport)
    summary = group.convergence[method]
    assert (summary.n_planned_rhs, summary.n_attempted_rhs, summary.n_converged) == (2, 0, 0)
    assert summary.iterations is None
    assert group.observations["system_id"].to_list() == [tag.system_id for tag in tags]
    assert group.observations["outcome"].to_list() == ["setup_failed"] * 2


def test_matrix_set_rhs_type_summary_allocates_setup_using_complete_matrix_membership(
    method,
) -> None:
    tags = (_tag(0, recipe=GAUSSIAN), _tag(1, recipe=SINE))
    setup = _setup(method)
    attempts = tuple(_attempt(tag, method) for tag in tags)

    reports = _classify(tags, (setup,), attempts, MatrixSetRhsTypeLevel())
    assert all(isinstance(report, RecipeProfileReport) for report in reports)

    rhs_type_reports = [report for report in reports if isinstance(report, RecipeProfileReport)]
    assert [report.summaries[0].allocated_setup_s for report in rhs_type_reports] == [1.0, 1.0]
    assert sum(
        report.summaries[0].allocated_setup_s for report in rhs_type_reports
    ) == pytest.approx(2.0)
    assert [report.observations["planned_rhs"].to_list() for report in rhs_type_reports] == [
        [2],
        [2],
    ]


def test_setup_failure_creates_asymmetric_censored_outcomes(method) -> None:
    tags = (_tag(0), _tag(1))
    reference = _setup(method)
    failed = replace(_setup(PreconditionerName("amg")), outcome=SetupFailed("failed"))
    attempts = tuple(_attempt(tag, method) for tag in tags)

    (group,) = _classify(
        tags,
        (reference, failed),
        attempts,
        MatrixSetRhsTypeLevel(),
        ConfiguredReference(method),
    )

    assert isinstance(group, RecipeProfileReport)
    comparison = speed_ratio_summary(group.observations, method, REFERENCE_METRIC).row(
        0, named=True
    )
    assert comparison["method"] == PreconditionerName("amg")
    assert (comparison["n_reference_only"], comparison["n_method_only"]) == (2, 0)


def test_unmeasured_group_retains_planned_membership(method) -> None:
    tags = (_tag(0), _tag(1))

    (group,) = _classify(tags, (), (), AllMatrixSetsLevel())

    assert isinstance(group, GroupSkippedAllCacheHit)
    assert (group.n_planned_rhs, group.n_matrices) == (2, 1)


def test_interrupted_method_is_retained_in_a_group_without_its_setup(method) -> None:
    """The workload plan, rather than observed setup events, defines method population."""
    reference = PreconditionerName("reference")
    candidate = PreconditionerName("candidate")
    tags = (_tag(0, recipe=ALPHA, matrix=0), _tag(0, recipe=BETA, matrix=1))
    setups = (_setup(reference, 0), _setup(reference, 1), _setup(candidate, 0))
    attempts = (
        _attempt(tags[0], reference),
        _attempt(tags[1], reference),
        _attempt(tags[0], candidate),
    )

    reports = _classify(
        tags,
        setups,
        attempts,
        MatrixSetRhsTypeLevel(),
        ConfiguredReference(reference),
        planned_by_method={reference: tags, candidate: tags},
    )

    by_recipe = {
        report.label.recipe: report for report in reports if isinstance(report, RecipeProfileReport)
    }
    interrupted = by_recipe[BETA]
    assert interrupted.convergence[candidate].n_attempted_rhs == 0
    summary = speed_ratio_summary(interrupted.observations, reference, REFERENCE_METRIC)
    assert summary.row(0, named=True)["method"] == candidate
    assert summary["n_reference_only"].to_list() == [1]


def test_matrix_set_rhs_type_and_matrix_set_labels_preserve_recipe_membership(method) -> None:
    tags = (_tag(0), _tag(0, recipe=SINE), _tag(0, dataset="other"))
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

    rhs_type = _classify(tags, setups, attempts, MatrixSetRhsTypeLevel())
    matrix_set = _classify(tags, setups, attempts, MatrixSetLevel())

    assert [group.label for group in rhs_type] == [
        MatrixSetRhsTypeLabel(DatasetId("ds"), GAUSSIAN),
        MatrixSetRhsTypeLabel(DatasetId("ds"), SINE),
        MatrixSetRhsTypeLabel(DatasetId("other"), GAUSSIAN),
    ]
    assert [group.label for group in matrix_set] == [
        MatrixSetLabel(DatasetId("ds"), (GAUSSIAN, SINE)),
        MatrixSetLabel(DatasetId("other"), (GAUSSIAN,)),
    ]
    assert all(isinstance(group, RecipeProfileReport) for group in rhs_type)
    assert [
        group.recipe_membership for group in rhs_type if isinstance(group, RecipeProfileReport)
    ] == [
        (GAUSSIAN,),
        (SINE,),
        (GAUSSIAN,),
    ]
    assert all(isinstance(group, WorkloadMixtureReport) for group in matrix_set)
    assert [
        group.recipe_membership for group in matrix_set if isinstance(group, WorkloadMixtureReport)
    ] == [(GAUSSIAN, SINE), (GAUSSIAN,)]

    assert [
        group.observations["system_id"].to_list()
        for group in rhs_type
        if isinstance(group, GroupAggregated)
    ] == [
        [tags[0].system_id],
        [tags[1].system_id],
        [tags[2].system_id],
    ]
    assert [
        group.observations["system_id"].to_list()
        for group in matrix_set
        if isinstance(group, GroupAggregated)
    ] == [
        [tags[0].system_id, tags[1].system_id],
        [tags[2].system_id],
    ]

    all_sets_report = _classify(tags, setups, attempts, AllMatrixSetsLevel())[0]
    assert isinstance(all_sets_report, WorkloadMixtureReport)
    assert all_sets_report.observations["system_id"].to_list() == [tag.system_id for tag in tags]
    assert all_sets_report.dataset_membership == (DatasetId("ds"), DatasetId("other"))
    assert all_sets_report.recipe_counts == (2, 1)


def test_removed_cost_models_are_absent_from_event_classification(method) -> None:
    tag = _tag(0)

    (group,) = _classify(
        (tag,), (_setup(method),), (_attempt(tag, method),), MatrixSetRhsTypeLevel()
    )

    assert isinstance(group, RecipeProfileReport)
    summary = group.convergence[method]
    for value in (group, summary):
        assert not hasattr(value, "ert_seconds")
        assert not hasattr(value, "recurring_s")
        assert not hasattr(value, "setup_sharing")


def test_workload_mixture_carries_the_pooled_observations_of_its_recipes(method) -> None:
    tags = (_tag(0), _tag(0, recipe=SINE))
    setup = _setup(method)
    attempts = tuple(_attempt(tag, method) for tag in tags)

    (group,) = _classify(tags, (setup,), attempts, AllMatrixSetsLevel())

    assert isinstance(group, WorkloadMixtureReport)
    assert group.observations.select("recipe", "system_id").rows() == [
        (recipe_key(tag.recipe), tag.system_id) for tag in tags
    ]
