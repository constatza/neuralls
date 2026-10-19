"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow

from neuralls.domain.solver.cost_metrics import (
    peak_memory_per_dof,
    setup_time_per_dof,
    time_per_dof_per_iteration,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.plot_style import plot_style_maps
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.ports import CostRecorder
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.reporting.artifacts import write_projected_toml
from neuralls.platform.reporting.plots import plot_projected_scaling_law, plot_system_time_bars
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
            as `domain.solver.metrics_table.report_level` returns.
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
        style = plot_style_maps(presentation.plot_styles)
        with tempfile.TemporaryDirectory() as tmp:
            # Group ids contain `/`, a path separator in artifact names.
            file_key = sanitize_metric_key_segment(group_id)
            stem = f"projected_scaling_law_{file_key}"
            plot_path = Path(tmp) / f"{stem}.png"
            toml_path = Path(tmp) / f"{stem}.toml"
            bars_path = Path(tmp) / f"system_time_bars_{file_key}.png"
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
            write_projected_toml(outcome.projected, toml_path)
            plot_system_time_bars(
                outcome.time_to_solution_by_system,
                save_path=bars_path,
                title=f"Time to solution per preconditioner — {presentation.subject}",
                subtitle=presentation.subtitle,
                labels=style.labels,
                families=style.families,
                color_keys=style.color_keys,
                marker_keys=style.marker_keys,
            )
            for artifact in (plot_path, toml_path, bars_path):
                mlflow.log_artifact(str(artifact), artifact_path=PROJECTED_ARTIFACT_DIR)


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
