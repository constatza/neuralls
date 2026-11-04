"""Fixtures shared by the platform storage tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from tests.measurement_support import NO_REFERENCE, NO_RESIDUAL, SINGLE_RUN
from tests.observation_support import (
    PLANNED,
    RHS_PER_MATRIX,
    SECOND_RECIPE_MATRICES,
    unbalanced_events,
    with_second_recipe,
)
from tests.presentation_support import PreparedMethodFactory
from tests.provenance_support import datasets_for, provenance_for
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    canonical_method_config,
)
from neuralls.domain.benchmarking.selection import ConfiguredReference
from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.models.cost import NotApplicable
from neuralls.domain.solver.models.figure_scope import FigureScope, RecipeScope
from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation, SolverTimingContext
from neuralls.domain.solver.models.labels import (
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
    render_group_id,
)
from neuralls.domain.solver.models.workload import MethodWorkloadSummary
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.observations import Outcome, observations_of_methods
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.benchmark_events import write_method_benchmark_artifacts
from neuralls.platform.reporting.figure_evidence import (
    MethodOutcomeCounts,
    NoInterval,
    SetupSharedOverRhs,
    TimingEvidence,
    TimingRepeats,
)
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureMethod,
    FigureSource,
    GroupAggregationRule,
    GroupFigureMetadata,
    MethodAggregationRule,
    MethodFigureMetadata,
    Statistic,
    TimingPolicy,
)
from neuralls.shared.rhs_recipe import SinusoidalRecipe
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
    PreconditionerName,
    RunId,
    SetupEventId,
    SystemId,
)

type WriteMethod = Callable[..., CompletedMethodWorkload]


@pytest.fixture
def write_method(
    prepared_method: PreparedMethodFactory, dataset_presentation: DatasetPresentation
) -> WriteMethod:
    """Write a complete one-system artifact set for a standard preconditioner."""

    def write(
        kind: PreconditionerType, root: Path, *, dataset: DatasetPresentation = dataset_presentation
    ) -> CompletedMethodWorkload:
        config = StandardPreconditionerConfig(type=kind)
        method = PreconditionerName(config.name)
        tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(0), 0, SystemId(f"{method}-0"))
        setup = MatrixSetupEvent(
            SetupEventId(f"{method}-setup"),
            tag.dataset,
            tag.matrix_id,
            method,
            1.0,
            (1.0,),
            MeasuredPeakMemory(1),
            SetupCompleted(),
        )
        attempt = SolveAttempt(
            setup.event_id,
            tag.system_id,
            tag.dataset,
            GAUSSIAN,
            tag.matrix_id,
            0,
            method,
            2.0,
            (2.0,),
            MeasuredPeakMemory(1),
            SolveConverged(2, 1e-8),
            NO_RESIDUAL,
            NO_REFERENCE,
            SINGLE_RUN,
        )
        workload = CompletedMethodWorkload(
            method=method,
            planned=(tag,),
            setup_events=(setup,),
            solve_attempts=(attempt,),
            diagnostics=(),
            generation_cost=NotApplicable(),
            workload_peak_memory=MeasuredPeakMemory(2),
            config_digest="sha256:" + "a" * 64,
            method_config=canonical_method_config(config),
            presentation=ExecutedMethod(prepared_method(str(method)), ()),
        )
        summary = summarize_method_workload(
            method, workload.planned, (setup,), (attempt,), workload.workload_peak_memory
        )
        write_method_benchmark_artifacts(
            workload,
            summary,
            root,
            systems=provenance_for(workload.planned),
            datasets=datasets_for(workload.planned, dataset),
        )
        return workload

    return write


CONTEXT = SolverTimingContext(
    rtol=1e-6,
    atol=1e-14,
    max_iterations=100,
    stopping_criterion="residual_norm",
    m_max=5,
    reference_precision_margin=2.0,
    timing_repeats=3,
    timing_warmup=1,
    timing_statistic="median",
    normalization="matrix",
)


@pytest.fixture
def figure_method(prepared_method: PreparedMethodFactory) -> FigureMethod:
    return FigureMethod(
        PreconditionerName("jacobi"),
        prepared_method("jacobi").detailed,
        "sha256:" + "a" * 64,
    )


@pytest.fixture
def metadata_of(
    figure_method: FigureMethod,
) -> Callable[[FigureScope], GroupFigureMetadata]:
    def build(scope: FigureScope) -> GroupFigureMetadata:
        return GroupFigureMetadata(
            kind=FigureKind.PERFORMANCE_PROFILE,
            metric_definition="amortized online seconds",
            metric=MetricId.ONLINE_S,
            population=Population.OWN,
            statistic=Statistic.PERFORMANCE_PROFILE,
            report_level=ReportLevelName.MATRIX_SET_RHS_TYPE,
            scope=scope,
            dataset_ids=(DatasetId("Dataset X"),),
            recipes=(SinusoidalRecipe(omega=6.28319),),
            n_matrices=1,
            n_planned_rhs=4,
            reference_selection=ConfiguredReference(PreconditionerName("jacobi")),
            failure_treatment=FailureTreatment.CENSORED,
            timing_policy=TimingPolicy.SETUP_PER_MATRIX_PLUS_SOLVE_PER_RHS,
            methods=(figure_method,),
            sources=(
                FigureSource(
                    RunId("method-run"),
                    ("method_workload/setup_events.parquet", "method_workload/summary.toml"),
                ),
            ),
            recipe_counts=(4,),
            aggregation_rule=GroupAggregationRule.PER_PLANNED_SYSTEM,
            recipe_weights=(1.0,),
            solver_contexts=(CONTEXT,),
            interval=NoInterval(),
            setup_sharing=SetupSharedOverRhs((4,)),
            outcome_counts=(
                MethodOutcomeCounts(
                    PreconditionerName("jacobi"),
                    {outcome: 4 if outcome is Outcome.CONVERGED else 0 for outcome in Outcome},
                ),
            ),
            timing=TimingEvidence((), TimingRepeats.REPEATED),
        )

    return build


@pytest.fixture
def method_metadata_of(
    figure_method: FigureMethod,
) -> Callable[[FigureScope], MethodFigureMetadata]:
    def build(scope: FigureScope) -> MethodFigureMetadata:
        return MethodFigureMetadata(
            kind=FigureKind.RESIDUAL_HISTORY,
            metric_definition="relative residual by iteration",
            report_level=ReportLevelName.SYSTEM,
            scope=scope,
            dataset_ids=(DatasetId("Dataset X"),),
            recipes=(SinusoidalRecipe(omega=6.28319),),
            n_matrices=1,
            n_planned_rhs=1,
            reference_selection=ConfiguredReference(PreconditionerName("jacobi")),
            timing_policy=TimingPolicy.DIAGNOSTIC_EXCLUDED_FROM_ONLINE,
            methods=(figure_method,),
            sources=(FigureSource(RunId("method-run"), ("method_workload/convergence.parquet",)),),
            recipe_counts=(1,),
            aggregation_rule=MethodAggregationRule.PER_METHOD_ON_SYSTEM,
            recipe_weights=(1.0,),
            solver_contexts=(CONTEXT,),
        )

    return build


AMG_TIMING_SPREAD = 0.25
"""The relative spread of the timing repeats of every ``amg`` solve in ``report_observations``."""


@pytest.fixture
def report_observations() -> pl.DataFrame:
    """The unbalanced three-method workload; ``amg`` solves carry a spread of repeated timings."""
    frames = {method: events.frames() for method, events in unbalanced_events().items()}
    return observations_of_methods(frames).with_columns(
        timing_spread=pl.when(pl.col("method") == "amg")
        .then(AMG_TIMING_SPREAD)
        .otherwise(pl.col("timing_spread"))
    )


@pytest.fixture
def workload_summaries() -> tuple[MethodWorkloadSummary, ...]:
    """The workload totals of the three methods of ``report_observations``."""
    return tuple(
        summarize_method_workload(
            events.method,
            PLANNED,
            events.setup_events,
            events.solve_attempts,
            MeasuredPeakMemory(1),
        )
        for events in unbalanced_events().values()
    )


@pytest.fixture
def profile_report(
    report_observations: pl.DataFrame, workload_summaries: tuple[MethodWorkloadSummary, ...]
) -> RecipeProfileReport:
    """One recipe over four matrices, compared against the configured reference ``jacobi``."""
    label = MatrixSetRhsTypeLabel(dataset=DatasetId("ds"), recipe=GAUSSIAN)
    return RecipeProfileReport(
        group_id=render_group_id(label),
        label=label,
        n_matrices=4,
        n_planned_rhs=len(PLANNED),
        observations=report_observations,
        summaries=workload_summaries,
        reference_selection=ConfiguredReference(PreconditionerName("jacobi")),
        recipe_membership=(GAUSSIAN,),
    )


@pytest.fixture
def mixture_report(
    profile_report: RecipeProfileReport,
) -> WorkloadMixtureReport:
    """The same workload pooled as a mixture of one named recipe."""
    return WorkloadMixtureReport(
        group_id=profile_report.group_id,
        label=MatrixSetLabel(DatasetId("ds"), (GAUSSIAN,)),
        n_matrices=profile_report.n_matrices,
        n_planned_rhs=profile_report.n_planned_rhs,
        observations=profile_report.observations,
        summaries=profile_report.summaries,
        reference_selection=profile_report.reference_selection,
        recipe_membership=(GAUSSIAN,),
        dataset_membership=(DatasetId("ds"),),
        recipe_counts=(profile_report.n_planned_rhs,),
    )


@pytest.fixture
def two_recipe_mixture_report(mixture_report: WorkloadMixtureReport) -> WorkloadMixtureReport:
    """The same workload as a matrix-set mixture of two recipes: matrices 3 and 4 use ``SINE``."""
    second = sum(RHS_PER_MATRIX[matrix] for matrix in SECOND_RECIPE_MATRICES)
    return replace(
        mixture_report,
        label=MatrixSetLabel(DatasetId("ds"), (GAUSSIAN, SINE)),
        observations=with_second_recipe(mixture_report.observations),
        recipe_membership=(GAUSSIAN, SINE),
        recipe_counts=(len(PLANNED) - second, second),
    )


@pytest.fixture
def repeated_timing_presentation(dataset_presentation: DatasetPresentation) -> GroupPresentation:
    """A group presentation whose systems were each timed three times."""
    return GroupPresentation(
        scope=RecipeScope(dataset_presentation, GAUSSIAN),
        plot_styles={},
        methods={},
        solver_contexts=(CONTEXT,),
    )


@pytest.fixture
def single_timing_presentation(
    repeated_timing_presentation: GroupPresentation,
) -> GroupPresentation:
    """A group presentation whose systems were each timed once."""
    return replace(
        repeated_timing_presentation, solver_contexts=(replace(CONTEXT, timing_repeats=1),)
    )
