"""Aggregate reporting consumes only completed canonical method workloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.presentation import GroupPresentation, SolverTimingContext
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
    PreconditionerName,
    SetupEventId,
    SystemId,
)


@dataclass(slots=True)
class RecordingSink:
    outcomes: list[GroupAggregated] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        del presentation
        self.outcomes.append(outcome)


def test_group_reporting_reduces_completed_workload_events() -> None:
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
    )
    sink = RecordingSink()

    report_all_levels((workload,), sink)

    assert [outcome.n_planned_rhs for outcome in sink.outcomes] == [1, 1, 1]


def test_group_reporting_propagates_method_plot_styles() -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("styled"))
    event_id = SetupEventId("identity:ds:1")
    style = PreconditionerPlotStyle("Configured Identity", PreconditionerType.IDENTITY)
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
        plot_styles={str(method): style},
    )

    class PresentationSink:
        def __init__(self) -> None:
            self.presentations: list[GroupPresentation] = []

        def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
            del outcome
            self.presentations.append(presentation)

    sink = PresentationSink()
    report_all_levels((workload,), sink)

    assert sink.presentations[0].plot_styles[str(method)] == style


def test_mixture_presentations_keep_only_the_short_scope() -> None:
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
    )

    class PresentationSink:
        def __init__(self) -> None:
            self.presentations: list[GroupPresentation] = []

        def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
            del outcome
            self.presentations.append(presentation)

    sink = PresentationSink()
    report_all_levels((workload,), sink)

    assert [presentation.subtitle for presentation in sink.presentations] == [
        "Dataset ds - gaussian",
        "Dataset ds",
        "All datasets",
    ]


def test_group_reporting_propagates_actual_solver_timing_context() -> None:
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
    report_all_levels((workload,), sink, contexts_by_system={tag.system_id: context})

    assert [presentation.solver_contexts for presentation in sink.presentations] == [
        (context,),
        (context,),
        (context,),
    ]


def test_group_reporting_disambiguates_duplicate_configured_labels() -> None:
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(1), 0, SystemId("labels"))

    def workload(name: str) -> CompletedMethodWorkload:
        method = PreconditionerName(name)
        return CompletedMethodWorkload(
            method=method,
            planned=(tag,),
            setup_events=(),
            solve_attempts=(),
            diagnostics=(),
            generation_cost=None,
            workload_peak_memory=MeasuredPeakMemory(1),
            config_digest="sha256:" + name[-1] * 64,
            method_config={"name": name},
            plot_styles={name: PreconditionerPlotStyle("Jacobi", PreconditionerType.JACOBI)},
        )

    class PresentationSink:
        def __init__(self) -> None:
            self.presentations: list[GroupPresentation] = []

        def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
            del outcome
            self.presentations.append(presentation)

    sink = PresentationSink()
    report_all_levels((workload("jacobi_a"), workload("jacobi_b")), sink)

    assert sink.presentations[0].plot_styles["jacobi_a"].label == "Jacobi [jacobi_a]"
    assert sink.presentations[0].plot_styles["jacobi_b"].label == "Jacobi [jacobi_b]"
