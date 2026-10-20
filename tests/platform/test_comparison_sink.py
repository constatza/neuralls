"""`MLflowGroupReportSink`: counts, ratios, and artifacts for one event-backed group."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.labels import FineLabel, render_group_id
from neuralls.domain.solver.models.outcomes import Failed, Solved
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.models.statistics import (
    Interval,
    IterationStats,
    MethodComparison,
    PairedComparison,
)
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.tracking import comparison_tracking
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    MLflowGroupReportSink,
)
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SystemId

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
def group() -> GroupAggregated:
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
    return GroupAggregated(
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
                    wilcoxon_p=0.25,
                ),
            )
        },
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


def _key(metric: str, group: GroupAggregated, name: str) -> str:
    return comparison_tracking._group_metric_key(metric, group.group_id, name)


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
    assert logged_metrics[_key("wilcoxon_p", group, "jacobi")] == 0.25


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
            "system_time_scatter",
            "paired_time_scatter",
            "speedup_cdf",
            "performance_profile",
            "ecdf_iterations",
            "ecdf_time",
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

    assert not [name for name in logged_artifacts if name.startswith("speedup_cdf")]
    assert not [key for key in logged_metrics if key.startswith("speed_ratio")]
