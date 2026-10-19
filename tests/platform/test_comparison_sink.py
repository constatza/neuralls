"""`MLflowGroupReportSink`: the metrics and artifacts one group report logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.labels import FineLabel, render_group_id
from neuralls.domain.solver.models.outcomes import Failed, Solved
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.models.statistics import (
    BreakEvenAt,
    Interval,
    IterationStats,
    MethodComparison,
    PairedComparison,
)
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.tracking import comparison_tracking
from neuralls.platform.tracking.comparison_tracking import (
    PROJECTED_ARTIFACT_DIR,
    MLflowGroupReportSink,
)
from neuralls.shared.types import DatasetId

type LoggedMetrics = dict[str, float]


@pytest.fixture
def logged_metrics(monkeypatch: pytest.MonkeyPatch) -> LoggedMetrics:
    metrics: LoggedMetrics = {}
    monkeypatch.setattr(comparison_tracking.mlflow, "log_metric", metrics.__setitem__)
    return metrics


@pytest.fixture
def logged_params(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    params: dict[str, str] = {}
    monkeypatch.setattr(comparison_tracking.mlflow, "log_param", params.__setitem__)
    return params


@pytest.fixture
def logged_artifacts(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[str, bool]]:
    """Logged artifact file name -> (artifact directory, file existed when logged)."""
    artifacts: dict[str, tuple[str, bool]] = {}

    def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        artifacts[Path(local_path).name] = (str(artifact_path), Path(local_path).exists())

    monkeypatch.setattr(comparison_tracking.mlflow, "log_artifact", log_artifact)
    return artifacts


def _projected(name: str) -> ProjectedComparisonResult:
    return ProjectedComparisonResult(
        preconditioner_name=name,
        n_systems=4,
        n_converged=3,
        setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=False),
        setup_cost_total_seconds=4.0,
        generation_cost_total_seconds=None,
        total_solve_time_seconds=8.0,
        total_iterations=40,
    )


@pytest.fixture
def group() -> GroupAggregated:
    label = FineLabel(dataset=DatasetId("ds"), recipe="gaussian")
    amg = {f"s{i}": Solved(5 + i, 1.0 + i) for i in range(4)}
    amg["s3"] = Failed()
    jacobi = {"s0": Solved(50, 2.0), "s1": Solved(60, 2.5), "s2": Solved(70, 9.0), "s3": Failed()}
    return GroupAggregated(
        group_id=render_group_id(label),
        label=label,
        projected={"amg": _projected("amg")},
        n_aggregated=4,
        n_matrices=1,
        n_in_group=4,
        convergence={
            "amg": ConvergenceSummary(
                n_systems=4,
                n_converged=3,
                ert_seconds=3.5,
                iterations=IterationStats(mean=6.0, median=6.0, std=1.0),
                ert_interval=Interval(low=3.0, high=4.0, confidence=0.95),
            ),
            "never": ConvergenceSummary(n_systems=4, n_converged=0, ert_seconds=None),
        },
        outcomes={"amg": amg, "jacobi": jacobi, "never": {f"s{i}": Failed() for i in range(4)}},
        comparisons={
            "jacobi": MethodComparison(
                reference="amg",
                break_even=BreakEvenAt(n_systems=7.5),
                paired=PairedComparison(
                    n_both_solved=3,
                    speedups=(0.5, 0.4, 0.2),
                    median_speedup=0.4,
                    median_speedup_ci=Interval(low=0.2, high=0.5, confidence=0.95),
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
    return comparison_tracking._projected_metric_key(metric, group.group_id, name)


def test_sink_logs_convergence_and_comparison_metrics(
    group, presentation, logged_metrics, logged_params, logged_artifacts
) -> None:
    MLflowGroupReportSink().report(group, presentation=presentation)

    assert logged_metrics[_key("n_converged", group, "amg")] == 3
    assert logged_metrics[_key("success_rate", group, "amg")] == pytest.approx(0.75)
    assert logged_metrics[_key("ert_s", group, "amg")] == 3.5
    assert logged_metrics[_key("ert_ci_low", group, "amg")] == 3.0
    assert logged_metrics[_key("iterations_median", group, "amg")] == 6.0
    assert logged_metrics[_key("break_even_n", group, "jacobi")] == 7.5
    assert logged_metrics[_key("speedup_median", group, "jacobi")] == 0.4
    assert logged_metrics[_key("speedup_ci_high", group, "jacobi")] == 0.5
    assert logged_metrics[_key("wilcoxon_p", group, "jacobi")] == 0.25


def test_never_converged_method_logs_counts_without_projected_or_ert_metrics(
    group, presentation, logged_metrics, logged_params, logged_artifacts
) -> None:
    MLflowGroupReportSink().report(group, presentation=presentation)

    assert logged_metrics[_key("success_rate", group, "never")] == 0.0
    assert not [k for k in logged_metrics if k.endswith("/never") and "ert_s" in k]
    assert _key("avg_setup_cost_s", group, "never") not in logged_metrics


def test_sink_logs_the_solved_system_solve_time_beside_the_per_system_one(
    group, presentation, logged_metrics, logged_params, logged_artifacts
) -> None:
    MLflowGroupReportSink().report(group, presentation=presentation)

    assert _key("avg_solve_time_per_solved_system_s", group, "amg") in logged_metrics
    assert _key("avg_solve_time_per_system_s", group, "amg") in logged_metrics


def test_sink_uploads_every_drawn_artifact_under_the_projected_directory(
    group, presentation, logged_metrics, logged_params, logged_artifacts
) -> None:
    MLflowGroupReportSink().report(group, presentation=presentation)

    key = comparison_tracking.sanitize_metric_key_segment(group.group_id)
    expected = {
        f"{stem}_{key}.{ext}"
        for stem, ext in [
            ("projected_scaling_law", "png"),
            ("projected_scaling_law", "toml"),
            ("system_time_scatter", "png"),
            ("paired_time_scatter", "png"),
            ("speedup_cdf", "png"),
            ("performance_profile", "png"),
            ("ecdf_iterations", "png"),
            ("ecdf_time", "png"),
        ]
    }
    assert set(logged_artifacts) == expected
    assert all(directory == PROJECTED_ARTIFACT_DIR for directory, _ in logged_artifacts.values())
    assert all(existed for _, existed in logged_artifacts.values())


def test_sink_skips_the_speedup_cdf_without_paired_comparisons(
    group, presentation, logged_metrics, logged_params, logged_artifacts
) -> None:
    without = GroupAggregated(
        group_id=group.group_id, label=group.label, projected=group.projected,
        n_aggregated=4, n_matrices=1, n_in_group=4, convergence=group.convergence,
        outcomes=group.outcomes,
    )  # fmt: skip

    MLflowGroupReportSink().report(without, presentation=presentation)

    assert not [name for name in logged_artifacts if name.startswith("speedup_cdf")]
    assert not [k for k in logged_metrics if k.startswith("break_even_n")]
