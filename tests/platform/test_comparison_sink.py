"""`MLflowReportSink`: counts, ratios, and artifacts for one event-backed group."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from tests.presentation_support import PreparedMethodFactory
from tests.recipe_support import ALPHA, BETA, GAUSSIAN, SINE

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.benchmarking.selection import ConfiguredReference, NoReferenceConfigured
from neuralls.domain.presentation.methods import ExecutedMethod, PreparedComparisonPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import SystemTag, UnknownPeakMemory
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.figure_scope import (
    AllDatasetsScope,
    FigureScope,
    MethodScope,
    RecipeScope,
    SystemScope,
)
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import (
    GroupMethodPresentation,
    GroupPresentation,
    SolverTimingContext,
)
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
    render_group_id,
)
from neuralls.domain.solver.models.outcomes import Failed, Solved
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.report_scope import System
from neuralls.domain.solver.models.statistics import (
    Interval,
    IterationStats,
    MethodComparison,
    PairedComparison,
)
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.domain.solver.models.workload import CompleteWorkloadSummary
from neuralls.domain.solver.reporting.event_frames import (
    CONVERGENCE_SCHEMA,
    SETUP_EVENT_SCHEMA,
    SOLVE_ATTEMPT_SCHEMA,
)
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    parse_preconditioner_config,
)
from neuralls.platform.reporting.figure_card import FigureCard, decode_figure_card
from neuralls.platform.reporting.figure_index import (
    FIGURE_INDEX_FILE,
    IncompleteFigureRecordsError,
)
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureMetadata,
    FigurePlan,
    SiblingTable,
    StagedFigure,
    TimingScenario,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.platform.tracking import comparison_tracking
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    METHOD_FIGURE_ARTIFACT_DIR,
    DuplicateFigureNameError,
    MLflowReportSink,
    UnknownStoredMethodError,
    _figure_plans,
)
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, GroupId, MatrixId, PreconditionerName, SystemId

type LoggedMetrics = dict[str, float]

_CONTEXT = SolverTimingContext(
    rtol=1e-6,
    atol=1e-14,
    max_iterations=20,
    stopping_criterion="residual_norm",
    m_max=5,
    reference_precision_margin=2.0,
    timing_repeats=1,
    timing_warmup=0,
    timing_statistic="median",
    normalization="none",
)


@pytest.fixture
def logged_metrics(monkeypatch: pytest.MonkeyPatch) -> LoggedMetrics:
    metrics: LoggedMetrics = {}
    monkeypatch.setattr(comparison_tracking.mlflow, "log_metric", metrics.__setitem__)
    return metrics


@pytest.fixture
def logged_artifacts(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[str, bool]]:
    """Logged artifact file name -> (artifact directory, file existed when logged)."""
    artifacts: dict[str, tuple[str, bool]] = {}

    def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        artifacts[Path(local_path).name] = (str(artifact_path), Path(local_path).exists())

    monkeypatch.setattr(comparison_tracking.mlflow, "log_artifact", log_artifact)
    return artifacts


@pytest.fixture
def logged_text(
    logged_artifacts: dict[str, tuple[str, bool]], monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """Logged TOML artifact file name -> its text when logged (cards and the figure index)."""
    texts: dict[str, str] = {}
    inner = comparison_tracking.mlflow.log_artifact

    def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        inner(local_path, artifact_path)
        if local_path.endswith(".toml"):
            texts[Path(local_path).name] = Path(local_path).read_text(encoding="utf-8")

    monkeypatch.setattr(comparison_tracking.mlflow, "log_artifact", log_artifact)
    return texts


def _cards(logged_text: dict[str, str]) -> dict[str, FigureCard]:
    return {
        name: decode_figure_card(tomllib.loads(text))
        for name, text in logged_text.items()
        if name != FIGURE_INDEX_FILE
    }


def _figures(logged_artifacts: dict[str, tuple[str, bool]]) -> set[str]:
    return {name for name in logged_artifacts if name.endswith(".png")}


def _solved(iterations: int, solve_s: float) -> Solved:
    return Solved(
        MatrixId(7),
        iterations,
        solve_s=solve_s,
        amortized_online_s=solve_s + 0.5,
    )


def _failed() -> Failed:
    return Failed(MatrixId(7), iterations=100, reason="iteration limit")


@pytest.fixture
def group() -> RecipeProfileReport:
    label = MatrixSetRhsTypeLabel(dataset=DatasetId("ds"), recipe=GAUSSIAN)
    amg = {
        SystemId("s0"): _solved(5, 1.0),
        SystemId("s1"): _solved(6, 2.0),
        SystemId("s2"): _solved(7, 4.0),
        SystemId("s3"): _failed(),
    }
    jacobi = {
        SystemId("s0"): _solved(50, 2.0),
        SystemId("s1"): _solved(60, 2.5),
        SystemId("s2"): _solved(70, 9.0),
        SystemId("s3"): _failed(),
    }
    never = {SystemId(f"s{i}"): _failed() for i in range(4)}
    return RecipeProfileReport(
        group_id=render_group_id(label),
        label=label,
        n_matrices=1,
        n_planned_rhs=4,
        convergence={
            PreconditionerName("amg"): ConvergenceSummary(
                n_planned_rhs=4,
                n_attempted_rhs=4,
                n_converged=3,
                iterations=IterationStats(mean=6.0, median=6.0, std=1.0),
            ),
            PreconditionerName("never"): ConvergenceSummary(
                n_planned_rhs=4,
                n_attempted_rhs=4,
                n_converged=0,
            ),
        },
        outcomes={
            PreconditionerName("amg"): amg,
            PreconditionerName("jacobi"): jacobi,
            PreconditionerName("never"): never,
        },
        comparisons={
            PreconditionerName("jacobi"): MethodComparison(
                reference=PreconditionerName("amg"),
                paired=PairedComparison(
                    speed_ratios=(0.5, 0.4, 0.2),
                    median_speed_ratio=0.4,
                    median_speed_ratio_ci=Interval(low=0.2, high=0.5, confidence=0.95),
                    n_both_solved=3,
                    n_reference_only=0,
                    n_method_only=0,
                ),
            )
        },
        summaries=(
            CompleteWorkloadSummary(
                method=PreconditionerName("amg"),
                n_planned_rhs=4,
                n_attempted_rhs=4,
                n_converged=3,
                setup_total_s=2.0,
                solve_total_s=7.0,
                workload_online_s=9.0,
                online_s_per_attempted_rhs=2.25,
                workload_peak_memory=UnknownPeakMemory(),
            ),
        ),
        reference_selection=ConfiguredReference(PreconditionerName("amg")),
        recipe_membership=(GAUSSIAN,),
    )


@pytest.fixture
def presentation(
    prepared_cohort: Callable[..., PreparedComparisonPresentation],
    dataset_presentation: DatasetPresentation,
) -> GroupPresentation:
    cohort = prepared_cohort("amg", "jacobi", "never")
    families = {
        "amg": PreconditionerType.AMG,
        "jacobi": PreconditionerType.JACOBI,
        "never": PreconditionerType.IDENTITY,
    }
    return GroupPresentation(
        scope=RecipeScope(dataset_presentation, GAUSSIAN),
        plot_styles={
            name: PreconditionerPlotStyle(cohort.method(PreconditionerName(name)), family)
            for name, family in families.items()
        },
        methods={
            PreconditionerName(name): GroupMethodPresentation(
                cohort.method(PreconditionerName(name)), ()
            )
            for name in families
        },
    )


@pytest.fixture
def stored_methods(
    prepared_cohort: Callable[..., PreparedComparisonPresentation],
) -> tuple[StoredMethod, ...]:
    """A stored run for each method of the group presentation."""
    cohort = prepared_cohort("amg", "jacobi", "never")
    tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(7), 0, SystemId("s0"))
    return tuple(
        StoredMethod(
            run_id=f"run-{name}",
            name=PreconditionerName(name),
            config_digest="sha256:" + "d" * 64,
            config=parse_preconditioner_config({"name": name, "type": "identity"}),
            presentation=ExecutedMethod(cohort.method(PreconditionerName(name)), ()),
            systems=(SystemProvenance(tag, _CONTEXT, NoReferenceConfigured()),),
            setup_events=pl.DataFrame(schema=SETUP_EVENT_SCHEMA),
            solve_attempts=pl.DataFrame(schema=SOLVE_ATTEMPT_SCHEMA),
            convergence=pl.DataFrame(schema=CONVERGENCE_SCHEMA),
        )
        for name in ("amg", "jacobi", "never")
    )


@pytest.fixture
def sink(notation: NotationPolicy, stored_methods: tuple[StoredMethod, ...]) -> MLflowReportSink:
    return MLflowReportSink(notation, "session-run", methods=stored_methods)


@pytest.fixture
def method_workload(prepared_method: PreparedMethodFactory) -> CompletedMethodWorkload:
    """Minimal method workload carrying the scientific presentation used by its plots."""
    method = PreconditionerName("network_a")
    prepared = prepared_method(str(method))
    return CompletedMethodWorkload(
        method=method,
        planned=(),
        setup_events=(),
        solve_attempts=(),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=UnknownPeakMemory(),
        config_digest="sha256:" + "b" * 64,
        method_config={"name": str(method)},
        presentation=ExecutedMethod(prepared, ()),
        plot_styles={str(method): PreconditionerPlotStyle(prepared, PreconditionerType.NEURAL)},
    )


@pytest.fixture(autouse=True)
def _no_system_metrics_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-system metrics are covered by the integration tree; none may start a real run here."""
    monkeypatch.setattr(comparison_tracking.mlflow, "log_metrics", lambda metrics: None)


@pytest.fixture(autouse=True)
def _log_into_the_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope-to-run nesting is covered by the integration tests; here a scope logs in place."""

    @contextmanager
    def in_place(sink: MLflowReportSink, scope: object) -> Generator[str]:
        yield f"run-of-{scope!r}"

    monkeypatch.setattr(MLflowReportSink, "_in_scope", in_place)


def _key(metric: str, group: GroupAggregated, name: str) -> str:
    return comparison_tracking.parent_run_metric_key(metric, name)


def _report_variant(
    group: RecipeProfileReport,
    report_type: type[RecipeProfileReport | WorkloadMixtureReport],
) -> RecipeProfileReport | WorkloadMixtureReport:
    membership = (GAUSSIAN,) if report_type is RecipeProfileReport else (GAUSSIAN, SINE)
    common = {
        "group_id": group.group_id,
        "label": (
            group.label
            if report_type is RecipeProfileReport
            else MatrixSetLabel(DatasetId("ds"), membership)
        ),
        "n_matrices": group.n_matrices,
        "n_planned_rhs": group.n_planned_rhs,
        "convergence": group.convergence,
        "summaries": group.summaries,
        "reference_selection": group.reference_selection,
        "recipe_membership": membership,
    }
    if report_type is RecipeProfileReport:
        return RecipeProfileReport(
            **common,
            outcomes=group.outcomes,
            comparisons=group.comparisons,
        )
    return WorkloadMixtureReport(**common)


_GROUP_FILE_STEM = "Test_problem_Gaussian_CG_Residuals_k_20_N_100_-_Gaussian"


def test_mixture_sink_omits_profiles_but_keeps_aggregate_bars(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    mixture = _report_variant(group, WorkloadMixtureReport)
    sink.report(mixture, presentation=presentation)

    key = _GROUP_FILE_STEM
    assert _figures(logged_artifacts) == {
        f"aggregate_bars_{key}.png",
        f"solve_time_bars_{key}.png",
        f"online_time_bars_{key}.png",
    }


def test_mixture_figure_plans_use_literal_population_and_metric_semantics(
    group: RecipeProfileReport,
) -> None:
    mixture = WorkloadMixtureReport(
        group_id=GroupId("all"),
        label=AllMatrixSetsLabel((ALPHA, BETA)),
        n_matrices=3,
        n_planned_rhs=4,
        convergence=group.convergence,
        summaries=group.summaries,
        reference_selection=group.reference_selection,
        recipe_membership=(ALPHA, BETA),
        dataset_membership=(DatasetId("first"), DatasetId("second")),
        recipe_counts=(1, 3),
    )
    context = _CONTEXT
    presentation = GroupPresentation(
        scope=AllDatasetsScope(),
        plot_styles={},
        methods={},
        solver_contexts=(context,),
    )

    plans = _figure_plans(mixture, presentation, "all", {})
    online = next(plan for plan in plans if plan.filename.startswith("online_time"))
    solve = next(plan for plan in plans if plan.filename.startswith("solve_time"))

    assert online.metadata.dataset_ids == ("first", "second")
    assert online.metadata.recipes == (ALPHA, BETA)
    assert online.metadata.recipe_counts == (1, 3)
    assert online.metadata.recipe_weights == (0.25, 0.75)
    assert online.metadata.metric_definition == "workload online seconds (setup + solve total)"
    assert online.metadata.scenario is TimingScenario.ONLINE
    assert online.metadata.failure_treatment is FailureTreatment.NONE
    assert online.metadata.solver_contexts == (context,)
    assert solve.metadata.metric_definition == "solve seconds (workload total)"
    assert solve.metadata.scenario is TimingScenario.SOLVE


def test_figure_plans_name_solve_and_online_ecdfs_by_scenario(
    group: RecipeProfileReport, dataset_presentation: DatasetPresentation
) -> None:
    """ECDF titles describe the explicit timing scenario being plotted."""
    presentation = GroupPresentation(
        scope=RecipeScope(dataset_presentation, GAUSSIAN), plot_styles={}, methods={}
    )
    plans = _figure_plans(group, presentation, "matrix_set_rhs_ds_gaussian", {})

    solve = next(
        plan
        for plan in plans
        if plan.metadata.kind.value == "ecdf_time"
        and plan.metadata.scenario is TimingScenario.SOLVE
    )
    online = next(
        plan
        for plan in plans
        if plan.metadata.kind.value == "ecdf_time"
        and plan.metadata.scenario is TimingScenario.ONLINE
    )

    scope = RecipeScope(dataset_presentation, GAUSSIAN)
    assert (solve.title, solve.metadata.scope) == ("Solve Time", scope)
    assert (online.title, online.metadata.scope) == ("Online Time", scope)


def test_paired_scatter_manifest_describes_axis_censored_pairs(
    group: RecipeProfileReport, dataset_presentation: DatasetPresentation
) -> None:
    """Paired scatter metadata covers asymmetric failures, unlike ratio CDFs."""
    presentation = GroupPresentation(
        scope=RecipeScope(dataset_presentation, GAUSSIAN), plot_styles={}, methods={}
    )
    plans = _figure_plans(group, presentation, "matrix_set_rhs_ds_gaussian", {})
    paired = next(plan for plan in plans if plan.metadata.kind.value == "paired_time")
    ratio = next(plan for plan in plans if plan.metadata.kind.value == "speed_ratio")

    assert paired.metadata.failure_treatment is FailureTreatment.CENSORED
    assert ratio.metadata.failure_treatment is FailureTreatment.OMITTED
    assert ratio.metadata.metric_definition != paired.metadata.metric_definition


def test_method_figures_are_uploaded_with_cards_naming_their_method_run(
    stored_methods: tuple[StoredMethod, ...],
    logged_artifacts: dict[str, tuple[str, bool]],
    logged_text: dict[str, str],
    sink: MLflowReportSink,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn: list[Path] = []
    monkeypatch.setattr(
        comparison_tracking, "_draw_method_figure", lambda path, *args: drawn.append(path)
    )
    jacobi = next(method for method in stored_methods if str(method.name) == "jacobi")

    sink.register_method_figures(datasets={})

    assert {path.name for path in drawn} == _figures(logged_artifacts)
    assert {directory for directory, _ in logged_artifacts.values()} == {METHOD_FIGURE_ARTIFACT_DIR}
    assert {name for name in logged_artifacts if name.endswith(".toml")} == {
        name.replace(".png", ".toml") for name in _figures(logged_artifacts)
    }
    card = _cards(logged_text)["residual_history_Method_Method_jacobi.toml"]
    metadata = card.metadata
    assert card.figure == "residual_history_Method_Method_jacobi.png"
    assert card.session_run_id == "session-run"
    assert metadata.scope == MethodScope(jacobi.presentation.prepared.detailed)
    assert metadata.dataset_ids == ("ds",)
    assert metadata.recipes == (GAUSSIAN,)
    assert metadata.solver_contexts == (_CONTEXT,)
    assert [(m.method_id, m.config_digest) for m in metadata.methods] == [
        (jacobi.name, jacobi.config_digest)
    ]
    assert [(source.run_id, source.artifact_paths) for source in metadata.sources] == [
        (
            "run-jacobi",
            (
                "method_workload/convergence.parquet",
                "method_workload/solve_attempts.parquet",
            ),
        )
    ]


def test_recipe_profile_sink_keeps_profile_ecdf_and_aggregate_bars(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    profile = _report_variant(group, RecipeProfileReport)
    sink.report(profile, presentation=presentation)

    key = _GROUP_FILE_STEM
    assert {
        f"aggregate_bars_{key}.png",
        f"performance_profile_solve_{key}.png",
        f"performance_profile_online_{key}.png",
    } <= _figures(logged_artifacts)


def test_sink_logs_explicit_counts_and_ratio_metrics(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    sink.report(group, presentation=presentation)

    assert logged_metrics[_key("n_planned_rhs", group, "amg")] == 4
    assert logged_metrics[_key("n_attempted_rhs", group, "amg")] == 4
    assert logged_metrics[_key("n_converged", group, "amg")] == 3
    assert logged_metrics[_key("success_rate", group, "amg")] == pytest.approx(0.75)
    assert logged_metrics[_key("iterations_median", group, "amg")] == 6.0
    assert logged_metrics[_key("n_both_solved", group, "jacobi")] == 3
    assert logged_metrics[_key("speed_ratio_median", group, "jacobi")] == 0.4
    assert logged_metrics[_key("speed_ratio_ci_high", group, "jacobi")] == 0.5
    assert not any("wilcoxon" in key for key in logged_metrics)


def test_never_converged_method_logs_only_counts(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    sink.report(group, presentation=presentation)

    assert logged_metrics[_key("success_rate", group, "never")] == 0.0
    assert not [key for key in logged_metrics if key.endswith("/never") and "iterations" in key]


def test_sink_uploads_only_failure_aware_group_artifacts(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    sink.report(group, presentation=presentation)

    key = _GROUP_FILE_STEM
    expected = {
        f"{stem}_{key}.png"
        for stem in (
            "aggregate_bars",
            "solve_time_bars",
            "online_time_bars",
            "paired_time_scatter",
            "speed_ratio_cdf",
            "performance_profile_solve",
            "performance_profile_online",
            "ecdf_iterations",
            "ecdf_time_solve",
            "ecdf_time_online",
        )
    }
    assert _figures(logged_artifacts) == expected
    assert set(logged_artifacts) == (
        expected
        | {name.replace(".png", ".toml") for name in expected}
        | {name.replace(".png", ".parquet") for name in expected}
    )
    assert all(directory == GROUP_REPORT_ARTIFACT_DIR for directory, _ in logged_artifacts.values())


def test_sink_skips_the_ratio_cdf_without_comparisons(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    sink.report(replace(group, comparisons={}), presentation=presentation)

    assert not [name for name in logged_artifacts if name.startswith("speed_ratio_cdf")]
    assert not [key for key in logged_metrics if key.startswith("speed_ratio")]


def test_group_figures_carry_cards_with_methods_parameters_and_sources(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_text: dict[str, str],
    sink: MLflowReportSink,
) -> None:
    sink.report(replace(group, comparisons={}), presentation=presentation)

    card = _cards(logged_text)[f"aggregate_bars_{_GROUP_FILE_STEM}.toml"]

    assert card.metadata.report_level.value == "matrix_set_rhs_type"
    assert [method.method_id for method in card.metadata.methods] == ["amg", "jacobi", "never"]
    assert [source.run_id for source in card.metadata.sources] == [
        "run-amg",
        "run-jacobi",
        "run-never",
    ]
    assert card.session_run_id == "session-run"
    assert card.table == SiblingTable(f"aggregate_bars_{_GROUP_FILE_STEM}.parquet")


@pytest.mark.usefixtures("logged_metrics", "logged_artifacts")
def test_group_naming_a_method_without_a_stored_run_is_rejected(
    group: GroupAggregated, presentation: GroupPresentation, notation: NotationPolicy
) -> None:
    with pytest.raises(UnknownStoredMethodError):
        MLflowReportSink(notation, "session-run").report(group, presentation=presentation)


def test_sink_finalizes_one_index_listing_exactly_the_logged_figures(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
    logged_text: dict[str, str],
    sink: MLflowReportSink,
) -> None:
    sink.report(group, presentation=presentation)
    sink.report(
        replace(_report_variant(group, RecipeProfileReport), group_id="other"),
        presentation=presentation,
    )
    sink.finalize()

    index = tomllib.loads(logged_text[FIGURE_INDEX_FILE])["figures"]
    assert logged_artifacts[FIGURE_INDEX_FILE] == ("None", True)
    assert {entry["artifact_path"] for entry in index} == {
        f"{GROUP_REPORT_ARTIFACT_DIR}/{name}" for name in _figures(logged_artifacts)
    }
    assert all(
        set(entry) == {"run_id", "artifact_path", "card", "kind", "level", "scope"}
        for entry in index
    )
    with pytest.raises(RuntimeError):
        sink.finalize()


@pytest.fixture
def system(dataset_presentation: DatasetPresentation) -> System:
    return System(DatasetId("ds"), GAUSSIAN, MatrixId(7), 0, SystemId("s0"))


@pytest.fixture
def system_figure(
    tmp_path: Path,
    metadata_of: Callable[[FigureScope], FigureMetadata],
    dataset_presentation: DatasetPresentation,
    system: System,
) -> StagedFigure:
    metadata = replace(
        metadata_of(SystemScope(dataset_presentation, system.recipe, system.matrix_id, 0)),
        kind=FigureKind.RESIDUAL_HISTORY,
        report_level=ReportLevelName.SYSTEM,
        n_matrices=1,
        n_planned_rhs=1,
    )
    plan = FigurePlan("residual_history.png", "Residual History", metadata)
    return StagedFigure(plan, tmp_path / plan.filename)


def test_system_figures_are_uploaded_with_cards_into_the_run_root(
    system: System,
    system_figure: StagedFigure,
    logged_artifacts: dict[str, tuple[str, bool]],
    logged_text: dict[str, str],
    sink: MLflowReportSink,
    dataset_presentation: DatasetPresentation,
) -> None:
    sink.report_system(system, {}, [system_figure])

    assert logged_artifacts == {
        "residual_history.png": ("None", False),
        "residual_history.toml": ("None", True),
    }
    card = _cards(logged_text)["residual_history.toml"]
    assert card.metadata.scope == SystemScope(dataset_presentation, GAUSSIAN, MatrixId(7), 0)
    assert card.metadata.report_level is ReportLevelName.SYSTEM


def test_one_stem_may_repeat_across_runs_but_not_within_one(
    system: System,
    system_figure: StagedFigure,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    other = replace(system, matrix_id=MatrixId(8))
    sink.report_system(system, {}, [system_figure])
    sink.report_system(other, {}, [system_figure])

    with pytest.raises(DuplicateFigureNameError):
        sink.report_system(other, {}, [system_figure])


def test_finalizing_with_a_figure_missing_its_card_raises(
    system: System,
    system_figure: StagedFigure,
    logged_artifacts: dict[str, tuple[str, bool]],
    sink: MLflowReportSink,
) -> None:
    sink.report_system(system, {}, [system_figure])
    sink._cards.clear()  # a recorded card lost: the invariant must catch it

    with pytest.raises(IncompleteFigureRecordsError):
        sink.finalize()
