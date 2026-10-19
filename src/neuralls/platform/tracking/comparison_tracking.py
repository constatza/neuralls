"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow

from neuralls.domain.benchmarking.costs import seconds
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
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.models.statistics import (
    AlwaysCheaper,
    BreakEvenAt,
    CheaperOnlyBelow,
    MethodComparison,
    NeverCheaper,
    PairedComparison,
)
from neuralls.domain.solver.ports import CostRecorder
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.reporting.artifacts import write_projected_toml
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    plot_ecdf_profile,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speedup_cdf,
    plot_time_scatter,
)
from neuralls.platform.reporting.plots import plot_projected_scaling_law
from neuralls.platform.tracking.mlflow import ensure_experiment, sanitize_metric_key_segment
from neuralls.shared.types import GroupId


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


def _projected_metric_key(metric: str, group_id: GroupId, preconditioner_name: str) -> str:
    """Build a projected-metric key, namespaced by comparison group and preconditioner.

    Mirrors ``parent_run_metric_key``'s single-``"/"``-separator convention,
    with one more segment: a case config can have several comparison groups,
    so the group id disambiguates which group a projection came from.
    """
    return (
        f"{metric}/{sanitize_metric_key_segment(group_id)}"
        f"/{sanitize_metric_key_segment(preconditioner_name)}"
    )


def log_projected_metrics(
    group_id: GroupId, projected: Mapping[str, ProjectedComparisonResult]
) -> None:
    """Log one comparison group's scaling-law projection to the open MLflow run.

    Logged onto whatever run is currently active (the session parent, since
    a projection aggregates across several sibling systems' own child
    runs, not any single one of them) — never opens a new run itself.

    Args:
        group_id: The comparison group's rendered id, used to namespace
            metric keys when a case has more than one group.
        projected: One `ProjectedComparisonResult` per preconditioner name,
            as `domain.solver.reporting.aggregation.report_level` returns.
    """
    for name, result in projected.items():
        mlflow.log_metric(_projected_metric_key("n_systems", group_id, name), result.n_systems)
        mlflow.log_metric(
            _projected_metric_key("avg_setup_cost_s", group_id, name),
            result.avg_setup_cost_seconds,
        )
        if result.avg_generation_cost_seconds is not None:
            mlflow.log_metric(
                _projected_metric_key("avg_generation_cost_s", group_id, name),
                result.avg_generation_cost_seconds,
            )
        mlflow.log_metric(
            _projected_metric_key("avg_solve_time_per_system_s", group_id, name),
            result.avg_solve_time_per_system_seconds,
        )
        mlflow.log_metric(
            _projected_metric_key("avg_solve_time_per_solved_system_s", group_id, name),
            result.avg_solve_time_per_solved_system_seconds,
        )
        if result.avg_iteration_time_seconds is not None:
            mlflow.log_metric(
                _projected_metric_key("avg_iteration_time_s", group_id, name),
                result.avg_iteration_time_seconds,
            )
        mlflow.log_param(
            _projected_metric_key("share_setup", group_id, name),
            str(result.setup_sharing.share_setup),
        )
        mlflow.log_param(
            _projected_metric_key("share_generation", group_id, name),
            str(result.setup_sharing.share_generation),
        )


def log_convergence_metrics(
    group_id: GroupId, convergence: Mapping[str, ConvergenceSummary]
) -> None:
    """Log each preconditioner's success rate and ERT beside its projected averages.

    ``avg_solve_time_per_system_s`` averages every run, failed ones at their
    measured cost, while ``avg_solve_time_per_solved_system_s`` divides the same
    total by the successes; the success rate belongs beside both. ERT
    (Hansen et al. 2016, arXiv:1605.03560) and its bootstrap interval are
    logged only when defined, as are the iteration statistics over converged runs.
    """
    for name, summary in convergence.items():
        mlflow.log_metric(_projected_metric_key("n_converged", group_id, name), summary.n_converged)
        mlflow.log_metric(
            _projected_metric_key("success_rate", group_id, name), summary.success_rate
        )
        if summary.ert_seconds is not None:
            mlflow.log_metric(_projected_metric_key("ert_s", group_id, name), summary.ert_seconds)
        if summary.ert_interval is not None:
            _log_optional_metrics(
                group_id,
                name,
                {
                    "ert_ci_low": summary.ert_interval.low,
                    "ert_ci_high": summary.ert_interval.high,
                },
            )
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
            mlflow.log_metric(_projected_metric_key(metric, group_id, preconditioner_name), value)


def _break_even_metrics(comparison: MethodComparison) -> dict[str, float | None]:
    """`break_even_n` for a finite break-even; the other outcomes have no count to log."""
    match comparison.break_even:
        case BreakEvenAt(n_systems=n_systems):
            return {"break_even_n": n_systems}
        case AlwaysCheaper() | NeverCheaper() | CheaperOnlyBelow():
            return {}


def _paired_metrics(paired: PairedComparison | None) -> dict[str, float | None]:
    """Median speedup, its interval and the Wilcoxon p-value of a paired comparison."""
    if paired is None:
        return {}
    interval = paired.median_speedup_ci
    return {
        "speedup_median": paired.median_speedup,
        "speedup_ci_low": interval.low if interval is not None else None,
        "speedup_ci_high": interval.high if interval is not None else None,
        "wilcoxon_p": paired.wilcoxon_p,
    }


def log_comparison_metrics(group_id: GroupId, comparisons: Mapping[str, MethodComparison]) -> None:
    """Log each method's break-even count, speedup over the reference and Wilcoxon p-value.

    Every metric is logged only when defined; the break-even count exists only
    for a finite crossing (`BreakEvenAt`). Speedup is reference time over method
    time, so values above 1 mean the method was faster.
    """
    for name, comparison in comparisons.items():
        _log_optional_metrics(
            group_id,
            name,
            {**_break_even_metrics(comparison), **_paired_metrics(comparison.paired)},
        )


class MLflowGroupReportSink:
    """``GroupReportSink`` that logs a group's projection into the open MLflow run.

    Logs the projected metrics, then uploads the scaling-law plot and its
    TOML numbers under ``projected_scaling_law``. Requires an open run: the
    composition layer only calls ``report`` when a projection exists, which
    only happens in a batch that executed something.
    """

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        """Log metrics, scaling plot, TOML numbers and per-system scatter for one group."""
        group_id = outcome.group_id
        log_projected_metrics(group_id, outcome.projected)
        log_convergence_metrics(group_id, outcome.convergence)
        log_comparison_metrics(group_id, outcome.comparisons)
        style = plot_style_maps(presentation.plot_styles)
        with tempfile.TemporaryDirectory() as tmp:
            # Group ids contain `/`, a path separator in artifact names.
            file_key = sanitize_metric_key_segment(group_id)
            stem = f"projected_scaling_law_{file_key}"
            plot_path = Path(tmp) / f"{stem}.png"
            toml_path = Path(tmp) / f"{stem}.toml"
            plot_projected_scaling_law(
                outcome.projected,
                save_path=plot_path,
                title=f"Projected cost per system vs. N — {presentation.subject}",
                subtitle=presentation.subtitle,
                labels=style.labels,
                families=style.families,
                color_keys=style.color_keys,
                marker_keys=style.marker_keys,
            )
            write_projected_toml(
                outcome.projected, outcome.convergence, outcome.comparisons, toml_path
            )
            outcome_plots = _write_outcome_plots(outcome, presentation, Path(tmp), file_key)
            for artifact in (plot_path, toml_path, *outcome_plots):
                mlflow.log_artifact(str(artifact), artifact_path=PROJECTED_ARTIFACT_DIR)


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
    comparisons: Mapping[str, MethodComparison],
) -> dict[str, PairedComparison]:
    """The paired comparisons of the methods that had any system solved alongside the reference."""
    return {
        name: comparison.paired
        for name, comparison in comparisons.items()
        if comparison.paired is not None
    }


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
            cost=seconds,
            xlabel="Recurring time per system: solve + per-system setup/generation [s]",
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


PROJECTED_ARTIFACT_DIR = "projected_scaling_law"
"""MLflow artifact directory holding every group's projected scaling-law files."""


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
