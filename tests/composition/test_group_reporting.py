"""Aggregate reporting consumes only completed canonical method workloads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from neuralls.composition.assignments.presentation_preflight import prepare_comparison_presentation
from neuralls.composition.comparison.method_data_origin import NoFitData
from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.composition.comparison.postprocessing import group_reporting
from neuralls.composition.comparison.postprocessing.group_reporting import report_all_levels
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
from neuralls.domain.solver.models.cost import NO_OFFLINE_COST
from neuralls.domain.solver.models.figure_scope import (
    AllDatasetsScope,
    DatasetScope,
    FigureScope,
    RecipeScope,
)
from neuralls.domain.solver.models.group_outcomes import GroupAggregated, GroupSkippedAllCacheHit
from neuralls.domain.solver.models.group_presentation import GroupPresentation, SolverTimingContext
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    GroupLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
    render_group_id,
)
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.reporting.cohorts import mixture_label
from neuralls.platform.config.models.coarsening import AggregationCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.reporting.figure_text import method_legend
from neuralls.shared.rhs_recipe import RhsRecipe, recipe_key
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.system_scaling import GershgorinScaling, RhsScaling
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
    NoFitDataEvidence,
    PreconditionerName,
    SetupEventId,
    SystemId,
)
from tests.fakes.split_store import InMemorySplitStore
from tests.measurement_support import NO_REFERENCE, NO_RESIDUAL, SINGLE_RUN
from tests.observation_support import observations_of_workloads
from tests.presentation_support import NO_DATASET_READS, PreparedMethodFactory
from tests.recipe_support import GAUSSIAN, SINE

type CapturedLog = Callable[[str], list[str]]


@dataclass(slots=True)
class RecordingSink:
    outcomes: list[GroupAggregated] = field(default_factory=list)
    presentations: list[GroupPresentation] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        self.outcomes.append(outcome)
        self.presentations.append(presentation)


def test_group_reporting_reduces_completed_workload_events(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    method = PreconditionerName("jacobi")
    tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(1), 0, SystemId("system"))
    event_id = SetupEventId("jacobi:ds:1")
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
                NO_RESIDUAL,
                NO_REFERENCE,
                SINGLE_RUN,
            ),
        ),
        diagnostics=(),
        offline_cost=NO_OFFLINE_COST,
        workload_peak_memory=MeasuredPeakMemory(20),
        config_digest="sha256:" + "0" * 64,
        method_config={"name": "jacobi"},
        data_evidence=NoFitDataEvidence(),
        presentation=ExecutedMethod(prepared_method("jacobi"), ()),
    )
    sink = RecordingSink()

    report_all_levels(
        (workload,),
        sink,
        observations=observations_of_workloads((workload,)),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
    )

    assert [outcome.n_planned_rhs for outcome in sink.outcomes] == [1]
    assert all(
        outcome.observations.select("method", "dataset").row(0) == ("jacobi", "ds")
        for outcome in sink.outcomes
    )


def test_group_reporting_propagates_method_plot_styles(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(1), 0, SystemId("styled"))
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
        offline_cost=NO_OFFLINE_COST,
        workload_peak_memory=MeasuredPeakMemory(10),
        config_digest="sha256:" + "1" * 64,
        method_config={"name": "identity"},
        data_evidence=NoFitDataEvidence(),
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
    report_all_levels(
        (workload,),
        sink,
        observations=observations_of_workloads((workload,)),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
    )

    assert sink.presentations[0].plot_styles[str(method)] == style


@pytest.fixture
def identity_workload(
    prepared_method: PreparedMethodFactory,
) -> Callable[[Sequence[SystemTag]], CompletedMethodWorkload]:
    """An identity workload that set up every matrix of ``tags`` and solved every system."""

    def build(tags: Sequence[SystemTag]) -> CompletedMethodWorkload:
        method = PreconditionerName("identity")
        events = {
            (tag.dataset, tag.matrix_id): MatrixSetupEvent(
                SetupEventId(f"identity:{tag.dataset}:{int(tag.matrix_id)}"),
                tag.dataset,
                tag.matrix_id,
                method,
                0.0,
                (0.0,),
                MeasuredPeakMemory(1),
                SetupCompleted(),
            )
            for tag in tags
        }
        return CompletedMethodWorkload(
            method=method,
            planned=tuple(tags),
            setup_events=tuple(events.values()),
            solve_attempts=tuple(
                SolveAttempt(
                    events[(tag.dataset, tag.matrix_id)].event_id,
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
                    NO_RESIDUAL,
                    NO_REFERENCE,
                    SINGLE_RUN,
                )
                for tag in tags
            ),
            diagnostics=(),
            offline_cost=NO_OFFLINE_COST,
            workload_peak_memory=MeasuredPeakMemory(1),
            config_digest="sha256:" + "2" * 64,
            method_config={"name": "identity"},
            data_evidence=NoFitDataEvidence(),
            presentation=ExecutedMethod(prepared_method("identity"), ()),
        )

    return build


@pytest.fixture
def two_matrix_sets(
    dataset_presentation: DatasetPresentation,
) -> dict[DatasetId, DatasetPresentation]:
    """The presentations of matrix sets ``ds`` and ``ds2``."""
    return {DatasetId("ds"): dataset_presentation, DatasetId("ds2"): dataset_presentation}


_DS, _DS2 = DatasetId("ds"), DatasetId("ds2")


def _short_scope(label: GroupLabel, datasets: dict[DatasetId, DatasetPresentation]) -> FigureScope:
    """The scope a group is presented with: a mixture keeps only its matrix set, or none."""
    match label:
        case MatrixSetRhsTypeLabel(dataset=dataset, recipe=recipe):
            return RecipeScope(datasets[dataset], recipe)
        case MatrixSetLabel(dataset=dataset):
            return DatasetScope(datasets[dataset])
        case AllMatrixSetsLabel():
            return AllDatasetsScope()


_MIX = f"equal-weight mixture of {recipe_key(GAUSSIAN)}, {recipe_key(SINE)}"


@pytest.mark.parametrize(
    ("recipes", "datasets", "labels", "mixture_labels"),
    [
        ((GAUSSIAN,), (_DS,), [MatrixSetRhsTypeLabel(_DS, GAUSSIAN)], []),
        (
            (GAUSSIAN, SINE),
            (_DS,),
            [
                MatrixSetRhsTypeLabel(_DS, GAUSSIAN),
                MatrixSetRhsTypeLabel(_DS, SINE),
                MatrixSetLabel(_DS, (GAUSSIAN, SINE)),
            ],
            [_MIX],
        ),
        (
            (GAUSSIAN,),
            (_DS, _DS2),
            [
                MatrixSetRhsTypeLabel(_DS, GAUSSIAN),
                MatrixSetRhsTypeLabel(_DS2, GAUSSIAN),
                AllMatrixSetsLabel((GAUSSIAN,)),
            ],
            ["pooled over 2 matrix sets"],
        ),
        (
            (GAUSSIAN, SINE),
            (_DS, _DS2),
            [
                MatrixSetRhsTypeLabel(_DS, GAUSSIAN),
                MatrixSetRhsTypeLabel(_DS, SINE),
                MatrixSetRhsTypeLabel(_DS2, GAUSSIAN),
                MatrixSetRhsTypeLabel(_DS2, SINE),
                MatrixSetLabel(_DS, (GAUSSIAN, SINE)),
                MatrixSetLabel(_DS2, (GAUSSIAN, SINE)),
                AllMatrixSetsLabel((GAUSSIAN, SINE)),
            ],
            [_MIX, _MIX, f"{_MIX}; pooled over 2 matrix sets"],
        ),
    ],
    ids=["one class", "two recipes", "two matrix sets", "two of both"],
)
def test_a_mixture_group_is_reported_only_when_it_pools_more_than_a_narrower_group(
    identity_workload: Callable[[Sequence[SystemTag]], CompletedMethodWorkload],
    two_matrix_sets: dict[DatasetId, DatasetPresentation],
    recipes: tuple[RhsRecipe, ...],
    datasets: tuple[DatasetId, ...],
    labels: list[GroupLabel],
    mixture_labels: list[str],
) -> None:
    tags = tuple(
        SystemTag(dataset, recipe, MatrixId(1), ordinal, SystemId(f"{dataset}-{ordinal}"))
        for dataset in datasets
        for ordinal, recipe in enumerate(recipes)
    )
    workload = identity_workload(tags)
    sink = RecordingSink()

    report_all_levels(
        (workload,),
        sink,
        observations=observations_of_workloads((workload,)),
        datasets=two_matrix_sets,
        notation=NotationPolicy.default(),
    )

    assert [outcome.label for outcome in sink.outcomes] == labels
    assert [presentation.scope for presentation in sink.presentations] == [
        _short_scope(label, two_matrix_sets) for label in labels
    ]
    assert [
        mixture_label(
            [recipe_key(recipe) for recipe in outcome.recipe_membership],
            len(outcome.dataset_membership),
        )
        for outcome in sink.outcomes
        if not isinstance(outcome.label, MatrixSetRhsTypeLabel)
    ] == mixture_labels


def test_group_reporting_propagates_actual_solver_timing_context(
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    method = PreconditionerName("identity")
    tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(1), 0, SystemId("context"))
    workload = CompletedMethodWorkload(
        method=method,
        planned=(tag,),
        setup_events=(),
        solve_attempts=(),
        diagnostics=(),
        offline_cost=NO_OFFLINE_COST,
        workload_peak_memory=MeasuredPeakMemory(1),
        config_digest="sha256:" + "3" * 64,
        method_config={"name": "identity"},
        data_evidence=NoFitDataEvidence(),
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
        system_scaling=GershgorinScaling(),
        rhs_scaling=RhsScaling.NONE,
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
        observations=observations_of_workloads((workload,)),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        contexts_by_system={tag.system_id: context},
    )

    assert [presentation.solver_contexts for presentation in sink.presentations] == [(context,)]


def test_group_reporting_keeps_aggregation_configuration_in_method_labels(
    dataset_presentation: DatasetPresentation,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(1), 0, SystemId("amg-labels"))
    configs = [
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.25)),
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.5)),
    ]
    cohort = prepare_comparison_presentation(
        configs,
        {config.name: NoFitData() for config in configs},
        ComparisonScopePresentation(dataset_presentation),
        NO_DATASET_READS,
        InMemorySplitStore(),
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
            offline_cost=NO_OFFLINE_COST,
            workload_peak_memory=MeasuredPeakMemory(1),
            config_digest="sha256:" + str(configs.index(config)) * 64,
            method_config={"name": config.name},
            data_evidence=NoFitDataEvidence(),
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
    workloads = tuple(workload(config) for config in configs)
    report_all_levels(
        workloads,
        sink,
        observations=observations_of_workloads(workloads),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
    )

    legend = method_legend(
        {name: method.compact for name, method in sink.presentations[0].methods.items()},
        NotationPolicy.default(),
    )
    assert [legend.label(config.name) for config in configs] == [
        "AMG (L=2, θ=0.25)",
        "AMG (L=2, θ=0.5)",
    ]


def test_a_skipped_group_is_named_by_its_scope_not_its_group_id(
    monkeypatch: pytest.MonkeyPatch,
    group_datasets: dict[DatasetId, DatasetPresentation],
    captured_log: CapturedLog,
) -> None:
    label = MatrixSetRhsTypeLabel(dataset=DatasetId("ds"), recipe=GAUSSIAN)
    skipped = GroupSkippedAllCacheHit(
        group_id=render_group_id(label), label=label, n_planned_rhs=3, n_matrices=1
    )
    monkeypatch.setattr(group_reporting, "classify_level", lambda *args, **kwargs: (skipped,))
    messages = captured_log("WARNING")
    report_all_levels(
        (),
        RecordingSink(),
        observations=observations_of_workloads(()),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
    )

    assert messages[0] == (
        "Comparison matrix set: Test problem | CG Errors (k=20, N=100) - Gaussian: "
        "no canonical setup events among 3 planned systems."
    )
