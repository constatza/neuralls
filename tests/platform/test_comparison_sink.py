"""`MLflowGroupReportSink`: counts, ratios, and artifacts for one event-backed group."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.benchmarking.selection import ConfiguredReference
from neuralls.domain.solver.models.benchmark_events import SystemTag, UnknownPeakMemory
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.labels import (
    FineLabel,
    GlobalLabel,
    MediumLabel,
    render_group_id,
)
from neuralls.domain.solver.models.outcomes import Failed, Solved
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.presentation import GroupPresentation, SolverTimingContext
from neuralls.domain.solver.models.statistics import (
    Interval,
    IterationStats,
    MethodComparison,
    PairedComparison,
)
from neuralls.domain.solver.models.workload import CompleteWorkloadSummary
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.reporting.figure_metadata import FailureTreatment, TimingScenario
from neuralls.platform.tracking import comparison_tracking
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    MLflowGroupReportSink,
    _figure_plans,
)
from neuralls.shared.types import DatasetId, GroupId, MatrixId, PreconditionerName, SystemId

type LoggedMetrics = dict[str, float]


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
    label = FineLabel(dataset=DatasetId("ds"), recipe="gaussian")
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
        recipe_membership=("gaussian",),
    )


@pytest.fixture
def presentation() -> GroupPresentation:
    return GroupPresentation(
        subject="ds, gaussian",
        subtitle="fine, 4 systems",
        plot_styles={
            "amg": PreconditionerPlotStyle(label="AMG", family=PreconditionerType.AMG),
            "jacobi": PreconditionerPlotStyle(label="Jacobi", family=PreconditionerType.JACOBI),
            "never": PreconditionerPlotStyle(label="None", family=PreconditionerType.IDENTITY),
        },
    )


@pytest.fixture
def method_workload() -> CompletedMethodWorkload:
    """Minimal method workload carrying the scientific label used by its plots."""
    method = PreconditionerName("network_a")
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
        plot_styles={
            str(method): PreconditionerPlotStyle(
                label="Neural (assignment=network_a)", family=PreconditionerType.NEURAL
            )
        },
    )


def _key(metric: str, group: GroupAggregated, name: str) -> str:
    return comparison_tracking._group_metric_key(metric, group.group_id, name)


def _report_variant(
    group: RecipeProfileReport,
    report_type: type[RecipeProfileReport | WorkloadMixtureReport],
) -> RecipeProfileReport | WorkloadMixtureReport:
    membership = (
        ("gaussian",)
        if report_type is RecipeProfileReport
        else ("gaussian", "sinusoidal ω=6.28319")
    )
    common = {
        "group_id": group.group_id,
        "label": (
            group.label
            if report_type is RecipeProfileReport
            else MediumLabel(DatasetId("ds"), membership)
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


def test_mixture_sink_omits_profiles_but_keeps_aggregate_bars(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
) -> None:
    mixture = _report_variant(group, WorkloadMixtureReport)
    MLflowGroupReportSink().report(mixture, presentation=presentation)

    key = comparison_tracking.sanitize_metric_key_segment(group.group_id)
    assert f"aggregate_bars_{key}.png" in logged_artifacts
    assert set(logged_artifacts) == {
        f"aggregate_bars_{key}.png",
        f"solve_time_bars_{key}.png",
        f"online_time_bars_{key}.png",
    }


def test_mixture_figure_plans_use_literal_population_and_metric_semantics(
    group: RecipeProfileReport,
) -> None:
    mixture = WorkloadMixtureReport(
        group_id=GroupId("global/dissolved"),
        label=GlobalLabel(("a", "b")),
        n_matrices=3,
        n_planned_rhs=4,
        convergence=group.convergence,
        summaries=group.summaries,
        reference_selection=group.reference_selection,
        recipe_membership=("a", "b"),
        dataset_membership=(DatasetId("first"), DatasetId("second")),
        recipe_counts=(1, 3),
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
    presentation = GroupPresentation(
        subject="All datasets",
        subtitle="All datasets",
        plot_styles={},
        solver_contexts=(context,),
    )

    plans = _figure_plans(mixture, presentation, "global_dissolved")
    online = next(plan for plan in plans if plan.filename.startswith("online_time"))
    solve = next(plan for plan in plans if plan.filename.startswith("solve_time"))

    assert online.metadata.datasets == ("first", "second")
    assert online.metadata.recipes == ("a", "b")
    assert online.metadata.recipe_counts == (1, 3)
    assert online.metadata.recipe_weights == (0.25, 0.75)
    assert online.metadata.metric_definition == "workload online seconds (setup + solve total)"
    assert online.metadata.scenario is TimingScenario.ONLINE
    assert online.metadata.failure_treatment is FailureTreatment.NONE
    assert online.metadata.solver_contexts == (context,)
    assert solve.metadata.metric_definition == "solve seconds (workload total)"
    assert solve.metadata.scenario is TimingScenario.SOLVE


def test_figure_plans_name_solve_and_online_ecdfs_by_scenario(
    group: RecipeProfileReport,
) -> None:
    """ECDF titles describe the explicit timing scenario being plotted."""
    presentation = GroupPresentation(
        subject="Dataset ds - gaussian", subtitle="Dataset ds", plot_styles={}
    )
    plans = _figure_plans(group, presentation, "fine_ds_gaussian")

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

    assert (solve.title, solve.subtitle) == ("Solve Time", "Dataset ds - gaussian")
    assert (online.title, online.subtitle) == ("Online Time", "Dataset ds - gaussian")


def test_paired_scatter_manifest_describes_axis_censored_pairs(
    group: RecipeProfileReport,
) -> None:
    """Paired scatter metadata covers asymmetric failures, unlike ratio CDFs."""
    presentation = GroupPresentation(
        subject="Dataset ds - gaussian", subtitle="Dataset ds", plot_styles={}
    )
    plans = _figure_plans(group, presentation, "fine_ds_gaussian")
    paired = next(plan for plan in plans if plan.metadata.kind.value == "paired_time")
    ratio = next(plan for plan in plans if plan.metadata.kind.value == "speed_ratio")

    assert paired.metadata.failure_treatment is FailureTreatment.CENSORED
    assert ratio.metadata.failure_treatment is FailureTreatment.OMITTED
    assert ratio.metadata.metric_definition != paired.metadata.metric_definition


def test_session_manifest_registers_uniquely_owned_method_figures() -> None:
    method = PreconditionerName("network_a")
    tag = SystemTag(DatasetId("ds"), "gaussian", MatrixId(7), 0, SystemId("s0"))
    workload = CompletedMethodWorkload(
        method=method,
        planned=(tag,),
        setup_events=(),
        solve_attempts=(),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=UnknownPeakMemory(),
        config_digest="sha256:" + "a" * 64,
        method_config={"name": str(method)},
        plot_styles={
            str(method): PreconditionerPlotStyle(
                label="Neural (assignment=network_a)", family=PreconditionerType.NEURAL
            )
        },
    )
    context = SolverTimingContext(
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
    sink = MLflowGroupReportSink()

    sink.register_method_figures((workload,), contexts_by_system={tag.system_id: context})

    assert set(sink.manifest.entries) == {
        "residual_history_network_a.png",
        "energy_history_network_a.png",
    }
    metadata = sink.manifest.entries["residual_history_network_a.png"]
    assert metadata.scope == "Method Neural (assignment=network_a)"
    assert metadata.datasets == ("ds",)
    assert metadata.recipes == ("gaussian",)
    assert metadata.solver_contexts == (context,)
    assert metadata.sources == (
        "comparison_method[method=network_a]/method_workload/method_diagnostics.json",
        "comparison_method[method=network_a]/method_workload/solve_attempts.parquet",
    )


def test_method_diagnostics_use_manifest_method_scope(
    tmp_path: Path,
    method_workload: CompletedMethodWorkload,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production history rendering receives the same concise scope as its manifest."""
    calls: list[str] = []
    monkeypatch.setattr(
        comparison_tracking,
        "plot_convergence_comparison",
        lambda *args, **kwargs: calls.append(kwargs["subtitle"]),
    )
    monkeypatch.setattr(
        comparison_tracking,
        "plot_error_convergence_comparison",
        lambda *args, **kwargs: calls.append(kwargs["subtitle"]),
    )

    comparison_tracking._write_method_diagnostics(tmp_path, method_workload)

    assert calls == [
        "Method Neural (assignment=network_a)",
        "Method Neural (assignment=network_a)",
    ]


def test_recipe_profile_sink_keeps_profile_ecdf_and_aggregate_bars(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
) -> None:
    profile = _report_variant(group, RecipeProfileReport)
    MLflowGroupReportSink().report(profile, presentation=presentation)

    key = comparison_tracking.sanitize_metric_key_segment(group.group_id)
    assert f"aggregate_bars_{key}.png" in logged_artifacts
    assert f"performance_profile_solve_{key}.png" in logged_artifacts
    assert f"performance_profile_online_{key}.png" in logged_artifacts


def test_sink_logs_explicit_counts_and_ratio_metrics(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
) -> None:
    MLflowGroupReportSink().report(group, presentation=presentation)

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
) -> None:
    MLflowGroupReportSink().report(group, presentation=presentation)

    assert logged_metrics[_key("success_rate", group, "never")] == 0.0
    assert not [key for key in logged_metrics if key.endswith("/never") and "iterations" in key]


def test_sink_uploads_only_failure_aware_group_artifacts(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
) -> None:
    MLflowGroupReportSink().report(group, presentation=presentation)

    key = comparison_tracking.sanitize_metric_key_segment(group.group_id)
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
    assert set(logged_artifacts) == expected
    assert all(directory == GROUP_REPORT_ARTIFACT_DIR for directory, _ in logged_artifacts.values())
    assert all(existed for _, existed in logged_artifacts.values())


def test_sink_skips_the_ratio_cdf_without_comparisons(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
) -> None:
    MLflowGroupReportSink().report(replace(group, comparisons={}), presentation=presentation)

    assert not [name for name in logged_artifacts if name.startswith("speed_ratio_cdf")]
    assert not [key for key in logged_metrics if key.startswith("speed_ratio")]


def test_sink_finalizes_one_manifest_for_logged_figures(
    group: GroupAggregated,
    presentation: GroupPresentation,
    logged_metrics: LoggedMetrics,
    logged_artifacts: dict[str, tuple[str, bool]],
) -> None:
    sink = MLflowGroupReportSink()
    sink.report(group, presentation=presentation)
    sink.report(
        replace(_report_variant(group, RecipeProfileReport), group_id="other"),
        presentation=presentation,
    )
    sink.finalize()

    assert [name for name in logged_artifacts if name == "figure_manifest.toml"] == [
        "figure_manifest.toml"
    ]
    with pytest.raises(RuntimeError):
        sink.finalize()
