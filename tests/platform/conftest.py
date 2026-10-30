"""Fixtures shared by the platform storage tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from tests.presentation_support import PreparedMethodFactory
from tests.provenance_support import datasets_for, provenance_for
from tests.recipe_support import GAUSSIAN

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
from neuralls.domain.solver.models.figure_scope import FigureScope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.benchmark_events import write_method_benchmark_artifacts
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureMethod,
    FigureSource,
    GroupAggregationRule,
    GroupFigureMetadata,
    MethodAggregationRule,
    MethodFigureMetadata,
    TimingPolicy,
    TimingScenario,
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
        )
        workload = CompletedMethodWorkload(
            method=method,
            planned=(tag,),
            setup_events=(setup,),
            solve_attempts=(attempt,),
            diagnostics=(),
            generation_cost=None,
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
            scenario=TimingScenario.ONLINE,
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
