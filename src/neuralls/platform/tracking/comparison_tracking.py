"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow

from neuralls.domain.benchmarking.costs import amortized_online_seconds
from neuralls.domain.solver.cost_metrics import (
    peak_memory_per_dof,
    setup_time_per_dof,
    time_per_dof_per_iteration,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.plot_style import PlotStyleMaps, plot_style_maps
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.models.statistics import MethodComparison, PairedComparison
from neuralls.domain.solver.ports import CostRecorder
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    plot_ecdf_profile,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speedup_cdf,
    plot_time_scatter,
)
from neuralls.platform.tracking.mlflow import ensure_experiment, sanitize_metric_key_segment
from neuralls.shared.types import GroupId, PreconditionerName


def setup_comparison_tracking(
    tracking_uri: str,
    artifact_location: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow tracking for comparison runs."""
    resolved_experiment_name = experiment_name or ExperimentNamesConfig().comparison
    ensure_experiment(
        resolved_experiment_name,
        MlflowPaths(tracking_uri=tracking_uri, artifact_uri=artifact_location),
    )
    mlflow.set_experiment(resolved_experiment_name)


def log_comparison_artifact_uri() -> None:
    """Log the current comparison run artifact URI."""
    mlflow.log_param("artifact_uri", mlflow.get_artifact_uri())


def log_skipped_preconditioners(warnings: Sequence[str]) -> None:
    """Log skipped-preconditioner count and reasons so drops are visible on the run.

    Each warning already carries the preconditioner name and failure reason
    (see ``resolve_preconditioner_models_with_warnings``). The count alone is not
    enough to notice a silently thinner comparison, so the full messages are
    logged as a text artifact next to the comparison plot.
    """
    if warnings:
        mlflow.log_param("skipped_preconditioners", str(len(warnings)))
        mlflow.log_text("\n".join(warnings), "skipped_preconditioners.txt")


def parent_run_metric_key(metric: str, preconditioner_name: str) -> str:
    """Build the parent-run key for one preconditioner's metric.

    MLflow's metric namespace is flat strings only — no structured/tuple
    keys — so every per-preconditioner metric on the parent run is named
    ``"<metric>/<preconditioner>"`` (MLflow's own UI renders ``"/"`` as a
    tree, which is why this convention exists at all). This is the single
    place that builds such a key: callers (including tests asserting on
    logged metric names) should use this instead of hand-reconstructing the
    separator themselves, so the convention can't silently drift between
    production code and its tests.

    Args:
        metric: Bare metric name, e.g. ``"iterations"``, ``"setup_time_s"``.
        preconditioner_name: Preconditioner's result key (sanitized here).

    Returns:
        The full parent-run metric key, e.g. ``"iterations/jacobi"``.
    """
    return f"{metric}/{sanitize_metric_key_segment(preconditioner_name)}"


def log_comparison_result_metrics(
    result: ComparisonResult,
    *,
    child_run_tags: Mapping[str, Mapping[str, str]],
) -> None:
    """Log parent-run metrics and nested child-run metrics for a comparison result."""
    system_size = result.matrix_shape[0] if result.matrix_shape is not None else None

    for name, cg in result.results.items():
        mlflow.log_metric(parent_run_metric_key("iterations", name), cg.iterations)
        mlflow.log_metric(parent_run_metric_key("final_residual", name), cg.residual)
        mlflow.log_metric(parent_run_metric_key("converged", name), int(cg.converged))
        for key, value in _cost_metrics(cg, system_size=system_size).items():
            mlflow.log_metric(parent_run_metric_key(key, name), value)
        if cg.generation_cost is not None:
            mlflow.log_param(
                parent_run_metric_key("generation_time_provenance", name),
                cg.generation_cost.provenance.value,
            )

        with mlflow.start_run(run_name=name, nested=True, tags=dict(child_run_tags[name])):
            for step, residual in enumerate(cg.residual_history_rel):
                mlflow.log_metric("residual", residual, step=step)
            for step, energy_error in enumerate(cg.error_history_a_rel or []):
                mlflow.log_metric("energy_error", energy_error, step=step)
            mlflow.log_metric("iterations", cg.iterations)
            mlflow.log_metric("final_residual", cg.residual)
            mlflow.log_metric("converged", int(cg.converged))
            for key, value in _cost_metrics(cg, system_size=system_size).items():
                mlflow.log_metric(key, value)
            if cg.generation_cost is not None:
                mlflow.log_param("generation_time_provenance", cg.generation_cost.provenance.value)

    if result.recommendations.overall_best is not None:
        mlflow.log_param("best_preconditioner", result.recommendations.overall_best.label)


def _group_metric_key(metric: str, group_id: GroupId, preconditioner_name: str) -> str:
    """Build a metric key namespaced by comparison group and method.

    Mirrors ``parent_run_metric_key``'s single-``"/"``-separator convention,
    with one more segment: a case config can have several comparison groups,
    so the group id disambiguates which report group produced the metric.
    """
    return (
        f"{metric}/{sanitize_metric_key_segment(group_id)}"
        f"/{sanitize_metric_key_segment(preconditioner_name)}"
    )


def log_convergence_metrics(
    group_id: GroupId, convergence: Mapping[PreconditionerName, ConvergenceSummary]
) -> None:
    """Log explicit planned, attempted, and converged counts."""
    for name, summary in convergence.items():
        mlflow.log_metric(_group_metric_key("n_planned_rhs", group_id, name), summary.n_planned_rhs)
        mlflow.log_metric(
            _group_metric_key("n_attempted_rhs", group_id, name), summary.n_attempted_rhs
        )
        mlflow.log_metric(_group_metric_key("n_converged", group_id, name), summary.n_converged)
        mlflow.log_metric(_group_metric_key("success_rate", group_id, name), summary.success_rate)
        if summary.iterations is not None:
            _log_optional_metrics(
                group_id,
                name,
                {
                    "iterations_mean": summary.iterations.mean,
                    "iterations_median": summary.iterations.median,
                    "iterations_std": summary.iterations.std,
                },
            )


def _log_optional_metrics(
    group_id: GroupId, preconditioner_name: str, metrics: Mapping[str, float | None]
) -> None:
    """Log each defined (non-`None`) metric under the group/preconditioner-namespaced key."""
    for metric, value in metrics.items():
        if value is not None:
            mlflow.log_metric(_group_metric_key(metric, group_id, preconditioner_name), value)


def _paired_metrics(paired: PairedComparison) -> dict[str, float | None]:
    """Failure counts, median ratio, interval, and paired Wilcoxon p-value."""
    interval = paired.median_speed_ratio_ci
    return {
        "n_both_solved": float(paired.n_both_solved),
        "n_reference_only": float(paired.n_reference_only),
        "n_method_only": float(paired.n_method_only),
        "speed_ratio_median": paired.median_speed_ratio,
        "speed_ratio_ci_low": interval.low if interval is not None else None,
        "speed_ratio_ci_high": interval.high if interval is not None else None,
        "wilcoxon_p": paired.wilcoxon_p,
    }


def log_comparison_metrics(
    group_id: GroupId, comparisons: Mapping[PreconditionerName, MethodComparison]
) -> None:
    """Log each method's failure-aware reference/method speed ratios."""
    for name, comparison in comparisons.items():
        _log_optional_metrics(
            group_id,
            name,
            _paired_metrics(comparison.paired),
        )


class MLflowGroupReportSink:
    """Temporary event-group sink retained until method-level reporting lands."""

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        """Log explicit counts and the supported failure-aware plots."""
        group_id = outcome.group_id
        log_convergence_metrics(group_id, outcome.convergence)
        log_comparison_metrics(group_id, outcome.comparisons)
        with tempfile.TemporaryDirectory() as tmp:
            file_key = sanitize_metric_key_segment(group_id)
            outcome_plots = _write_outcome_plots(outcome, presentation, Path(tmp), file_key)
            for artifact in outcome_plots:
                mlflow.log_artifact(str(artifact), artifact_path=GROUP_REPORT_ARTIFACT_DIR)


def _group_style(
    kind: str, presentation: GroupPresentation, plot_style: PlotStyleMaps
) -> GroupStyle:
    """The shared title and per-preconditioner styling of one outcome plot."""
    return GroupStyle(
        title=f"{kind} — {presentation.subject}",
        subtitle=presentation.subtitle,
        labels=plot_style.labels,
        families=plot_style.families,
        color_keys=plot_style.color_keys,
        marker_keys=plot_style.marker_keys,
    )


def _paired_by_method(
    comparisons: Mapping[PreconditionerName, MethodComparison],
) -> dict[PreconditionerName, PairedComparison]:
    """The paired comparisons of the methods that had any system solved alongside the reference."""
    return {name: comparison.paired for name, comparison in comparisons.items()}


def _write_profile_plots(
    outcome: GroupAggregated, presentation: GroupPresentation, directory: Path, file_key: str
) -> list[Path | None]:
    """Draw the performance profile and the iteration and time ECDFs of one group."""
    plot_style = plot_style_maps(presentation.plot_styles)
    return [
        plot_performance_profile(
            outcome.outcomes,
            style=_group_style("Performance profile (time)", presentation, plot_style),
            save_path=directory / f"performance_profile_{file_key}.png",
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=_group_style("ECDF (iterations)", presentation, plot_style),
            save_path=directory / f"ecdf_iterations_{file_key}.png",
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=_group_style("ECDF (time)", presentation, plot_style),
            cost=amortized_online_seconds,
            xlabel="Amortized online time per system [s]",
            save_path=directory / f"ecdf_time_{file_key}.png",
        ),
    ]


def _write_system_plots(
    outcome: GroupAggregated, presentation: GroupPresentation, directory: Path, file_key: str
) -> list[Path | None]:
    """Draw the per-system scatter, the paired scatter and the speedup CDF of one group."""
    plot_style = plot_style_maps(presentation.plot_styles)
    return [
        plot_time_scatter(
            outcome.outcomes,
            style=_group_style("Time to solution per system", presentation, plot_style),
            save_path=directory / f"system_time_scatter_{file_key}.png",
        ),
        plot_paired_time_scatter(
            outcome.outcomes,
            style=_group_style("Time to solution vs reference", presentation, plot_style),
            save_path=directory / f"paired_time_scatter_{file_key}.png",
        ),
        plot_speedup_cdf(
            _paired_by_method(outcome.comparisons),
            style=_group_style("Speedup over the reference", presentation, plot_style),
            save_path=directory / f"speedup_cdf_{file_key}.png",
        ),
    ]


def _write_outcome_plots(
    outcome: GroupAggregated, presentation: GroupPresentation, directory: Path, file_key: str
) -> tuple[Path, ...]:
    """Draw every per-system outcome plot of one group and return the files that were drawn."""
    drawn = [
        *_write_system_plots(outcome, presentation, directory, file_key),
        *_write_profile_plots(outcome, presentation, directory, file_key),
    ]
    return tuple(path for path in drawn if path is not None)


GROUP_REPORT_ARTIFACT_DIR = "group_reports"


class MLflowCostRecorder:
    """``CostRecorder`` implementation that logs into the currently open MLflow run.

    Thin wrapper reusing ``log_comparison_result_metrics``'s existing logic
    (not a duplicate) so composition-layer callers can depend on the
    ``domain.solver.ports.CostRecorder`` Protocol instead of importing this
    module's concrete MLflow function directly.
    """

    def record(
        self,
        result: ComparisonResult,
        *,
        child_run_tags: Mapping[str, Mapping[str, str]],
    ) -> None:
        """Log ``result``'s cost/outcome metrics into the open MLflow run."""
        log_comparison_result_metrics(result, child_run_tags=child_run_tags)


def create_cost_recorder() -> CostRecorder:
    """Return the default `CostRecorder`: MLflow, logging into the open run."""
    return MLflowCostRecorder()


def _cost_metrics(cg: CGComparisonResult, *, system_size: int | None) -> dict[str, float]:
    """Build the cost-metric dict for one preconditioner's result, omitting unset values.

    Shared between the parent-run (namespaced) and nested child-run (bare)
    logging loops in ``log_comparison_result_metrics`` so the metric set never
    drifts between the two.

    Args:
        cg: Comparison result to derive cost metrics from.
        system_size: Linear-system size ``n``, or ``None`` if unavailable —
            size-normalized metrics are skipped in that case.

    Returns:
        Mapping of metric name to value, containing only measured metrics.
    """
    metrics: dict[str, float] = {"setup_time_s": cg.setup_cost}
    if cg.generation_cost is not None:
        metrics["generation_time_s"] = cg.generation_cost.wall_time_seconds
    if cg.solve_time_seconds is not None:
        metrics["solve_time_s"] = cg.solve_time_seconds
    if cg.avg_iteration_time_seconds is not None:
        metrics["avg_iteration_time_s"] = cg.avg_iteration_time_seconds
    if cg.peak_memory_bytes is not None:
        metrics["peak_memory_bytes"] = float(cg.peak_memory_bytes)
    metrics.update(_normalized_cost_metrics(cg, system_size=system_size))
    return metrics


def _normalized_cost_metrics(
    cg: CGComparisonResult, *, system_size: int | None
) -> dict[str, float]:
    """Size-normalized cost metrics, empty when `system_size` is unavailable."""
    if system_size is None:
        return {}
    metrics: dict[str, float] = {}
    normalized_time = time_per_dof_per_iteration(cg, system_size=system_size)
    if normalized_time is not None:
        metrics["time_per_dof_per_iteration"] = normalized_time
    normalized_setup_time = setup_time_per_dof(cg, system_size=system_size)
    if normalized_setup_time is not None:
        metrics["setup_time_per_dof"] = normalized_setup_time
    normalized_memory = peak_memory_per_dof(cg, system_size=system_size)
    if normalized_memory is not None:
        metrics["peak_memory_per_dof"] = normalized_memory
    return metrics


def log_linear_system_params(result: ComparisonResult) -> None:
    """Log matrix/rhs shape and RHS provenance.

    A matrix may be paired with several different RHS sources (gaussian,
    sparse, raw, dataset, ...) across sibling ``[[comparisons]]`` entries,
    each its own MLflow run — ``rhs_source_kind`` disambiguates which one
    this particular run used.
    """
    if result.matrix_shape is not None:
        mlflow.log_param("matrix_rows", result.matrix_shape[0])
        mlflow.log_param("matrix_cols", result.matrix_shape[1])
    if result.rhs_shape is not None:
        mlflow.log_param("rhs_dim", result.rhs_shape[0])
    if result.rhs_source_kind is not None:
        mlflow.log_param("rhs_source_kind", str(result.rhs_source_kind))


def log_comparison_run_params(
    *,
    comp_run_id: str,
    comparison_id: str,
    comparison_display_name: str,
    comparison_config: str,
) -> None:
    """Log final metadata params to the active comparison MLflow run."""
    mlflow.log_param("comparison_config", comparison_config)
    mlflow.log_param("comparison_id", comparison_id)
    mlflow.log_param("comparison_display_name", comparison_display_name)
    mlflow.log_param("comp_run_id", comp_run_id)


def log_comparison_input_artifacts(artifact_dir: Path) -> None:
    """Upload the resolved comparison input system artifacts to MLflow."""
    mlflow.log_artifacts(str(artifact_dir), artifact_path="comparison_inputs")
