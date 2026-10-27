"""Aggregate reporting consumes only completed canonical method workloads."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from loguru import logger

from neuralls.composition.assignments.presentation_preflight import prepare_comparison_presentation
from neuralls.composition.comparison import group_reporting
from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.models.figure_scope import AllDatasetsScope, DatasetScope, RecipeScope
from neuralls.domain.solver.models.group_outcomes import GroupAggregated, GroupSkippedAllCacheHit
from neuralls.domain.solver.models.group_presentation import GroupPresentation, SolverTimingContext
from neuralls.domain.solver.models.labels import FineLabel, render_group_id
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.reporting.figure_text import method_legend
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)
from tests.presentation_support import PreparedMethodFactory


@dataclass(slots=True)
class RecordingSink:
    outcomes: list[GroupAggregated] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        del presentation
        self.outcomes.append(outcome)


def test_group_reporting_reduces_completed_workload_events(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("system"))
    event_id = SetupEventId("identity:ds:1")
    workload = CompletedMethodWorkload(
        method=method,
        planned=(tag,),
        setup_events=(
            MatrixSetupEvent(
                event_id,
                tag.dataset,
                tag.matrix_id,
                method,
                1.0,
                (1.0,),
                MeasuredPeakMemory(10),
                SetupCompleted(),
            ),
        ),
        solve_attempts=(
            SolveAttempt(
                event_id,
                tag.system_id,
                tag.dataset,
                tag.recipe,
                tag.matrix_id,
                tag.rhs_ordinal,
                method,
                2.0,
                (2.0,),
                MeasuredPeakMemory(20),
                SolveConverged(3, 1e-8),
            ),
        ),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(20),
        config_digest="sha256:" + "0" * 64,
        method_config={"name": "identity"},
        presentation=ExecutedMethod(prepared_method("identity"), ()),
    )
    sink = RecordingSink()

    report_all_levels((workload,), sink, datasets=group_datasets, notation=NotationPolicy.default())

    assert [outcome.n_planned_rhs for outcome in sink.outcomes] == [1, 1, 1]


def test_group_reporting_propagates_method_plot_styles(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("styled"))
    event_id = SetupEventId("identity:ds:1")
    style = PreconditionerPlotStyle(prepared_method("identity"), PreconditionerType.IDENTITY)
    workload = CompletedMethodWorkload(
        method=method,
        planned=(tag,),
        setup_events=(
            MatrixSetupEvent(
                event_id,
                tag.dataset,
                tag.matrix_id,
                method,
                1.0,
                (1.0,),
                MeasuredPeakMemory(10),
                SetupCompleted(),
            ),
        ),
        solve_attempts=(),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(10),
        config_digest="sha256:" + "1" * 64,
        method_config={"name": "identity"},
        presentation=ExecutedMethod(prepared_method("identity"), ()),
        plot_styles={str(method): style},
    )

    class PresentationSink:
        def __init__(self) -> None:
            self.presentations: list[GroupPresentation] = []

        def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
            del outcome
            self.presentations.append(presentation)

    sink = PresentationSink()
    report_all_levels((workload,), sink, datasets=group_datasets, notation=NotationPolicy.default())

    assert sink.presentations[0].plot_styles[str(method)] == style


def test_mixture_presentations_keep_only_the_short_scope(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("scope"))
    event_id = SetupEventId("identity:ds:1")
    workload = CompletedMethodWorkload(
        method=method,
        planned=(tag,),
        setup_events=(
            MatrixSetupEvent(
                event_id,
                tag.dataset,
                tag.matrix_id,
                method,
                0.0,
                (0.0,),
                MeasuredPeakMemory(1),
                SetupCompleted(),
            ),
        ),
        solve_attempts=(
            SolveAttempt(
                event_id,
                tag.system_id,
                tag.dataset,
                tag.recipe,
                tag.matrix_id,
                tag.rhs_ordinal,
                method,
                1.0,
                (1.0,),
                MeasuredPeakMemory(1),
                SolveConverged(1, 1e-8),
            ),
        ),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(1),
        config_digest="sha256:" + "2" * 64,
        method_config={"name": "identity"},
        presentation=ExecutedMethod(prepared_method("identity"), ()),
    )

    class PresentationSink:
        def __init__(self) -> None:
            self.presentations: list[GroupPresentation] = []

        def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
            del outcome
            self.presentations.append(presentation)

    sink = PresentationSink()
    report_all_levels((workload,), sink, datasets=group_datasets, notation=NotationPolicy.default())

    assert [presentation.scope for presentation in sink.presentations] == [
        RecipeScope(group_datasets[DatasetId("ds")], "gaussian"),
        DatasetScope(group_datasets[DatasetId("ds")]),
        AllDatasetsScope(),
    ]


def test_group_reporting_propagates_actual_solver_timing_context(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("context"))
    workload = CompletedMethodWorkload(
        method=method,
        planned=(tag,),
        setup_events=(),
        solve_attempts=(),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(1),
        config_digest="sha256:" + "3" * 64,
        method_config={"name": "identity"},
        presentation=ExecutedMethod(prepared_method("identity"), ()),
    )
    context = SolverTimingContext(
        rtol=1e-6,
        atol=1e-14,
        max_iterations=20,
        stopping_criterion="residual_norm",
        m_max=5,
        reference_precision_margin=2.0,
        timing_repeats=3,
        timing_warmup=1,
        timing_statistic="median",
        normalization="matrix",
    )

    class PresentationSink:
        def __init__(self) -> None:
            self.presentations: list[GroupPresentation] = []

        def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
            del outcome
            self.presentations.append(presentation)

    sink = PresentationSink()
    report_all_levels(
        (workload,),
        sink,
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        contexts_by_system={tag.system_id: context},
    )

    assert [presentation.solver_contexts for presentation in sink.presentations] == [
        (context,),
        (context,),
        (context,),
    ]


def test_group_reporting_keeps_aggregation_configuration_in_method_labels(
    dataset_presentation: DatasetPresentation,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("amg-labels"))
    configs = [
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.25)),
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.5)),
    ]
    cohort = prepare_comparison_presentation(
        configs, ComparisonScopePresentation(dataset_presentation)
    )

    def workload(config: AMGPreconditionerConfig) -> CompletedMethodWorkload:
        method = PreconditionerName(config.name)
        prepared = cohort.method(method)
        return CompletedMethodWorkload(
            method=method,
            planned=(tag,),
            setup_events=(),
            solve_attempts=(),
            diagnostics=(),
            generation_cost=None,
            workload_peak_memory=MeasuredPeakMemory(1),
            config_digest="sha256:" + str(configs.index(config)) * 64,
            method_config={"name": config.name},
            presentation=ExecutedMethod(prepared, ()),
            plot_styles={config.name: PreconditionerPlotStyle(prepared, PreconditionerType.AMG)},
        )

    class PresentationSink:
        def __init__(self) -> None:
            self.presentations: list[GroupPresentation] = []

        def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
            del outcome
            self.presentations.append(presentation)

    sink = PresentationSink()
    report_all_levels(
        tuple(workload(config) for config in configs),
        sink,
        datasets=group_datasets,
        notation=NotationPolicy.default(),
    )

    legend = method_legend(
        {name: method.compact for name, method in sink.presentations[0].methods.items()},
        NotationPolicy.default(),
    )
    assert [legend.label(config.name) for config in configs] == [
        "AMG (L=2, θ=0.25, ν₁=2, ν₂=2, ωₛ=auto)",
        "AMG (L=2, θ=0.5, ν₁=2, ν₂=2, ωₛ=auto)",
    ]


def test_a_skipped_group_is_named_by_its_scope_not_its_group_id(
    monkeypatch: pytest.MonkeyPatch,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    label = FineLabel(dataset=DatasetId("ds"), recipe="gaussian")
    skipped = GroupSkippedAllCacheHit(
        group_id=render_group_id(label), label=label, n_planned_rhs=3, n_matrices=1
    )
    monkeypatch.setattr(group_reporting, "classify_level", lambda *args, **kwargs: (skipped,))
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        report_all_levels(
            (), RecordingSink(), datasets=group_datasets, notation=NotationPolicy.default()
        )
    finally:
        logger.remove(sink_id)

    assert messages[0] == (
        "Test problem | Gaussian CG Residuals (k=20, N=100) - gaussian: "
        "no canonical setup events among 3 planned systems."
    )
