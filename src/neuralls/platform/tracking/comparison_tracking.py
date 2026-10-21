"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import mlflow

from neuralls.domain.benchmarking.costs import amortized_online_seconds, solve_seconds
from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    ReferenceUnavailable,
)
from neuralls.domain.solver.cost_metrics import (
    peak_memory_per_dof,
    setup_time_per_dof,
    time_per_dof_per_iteration,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.labels import GlobalLabel, MediumLabel
from neuralls.domain.solver.models.plot_style import PlotStyleMaps, plot_style_maps
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.models.statistics import MethodComparison, PairedComparison
from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary
from neuralls.domain.solver.ports import CostRecorder
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.reporting.benchmark_events import (
    BENCHMARK_EVENT_SCHEMA_VERSION,
    write_method_benchmark_artifacts,
)
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureManifest,
    FigureMetadata,
    FigurePlan,
    TimingScenario,
    make_plan,
    write_figure_manifest,
)
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    plot_aggregate_bars,
    plot_ecdf_profile,
    plot_online_time_bars,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speed_ratio_cdf,
)
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
    plot_error_convergence_comparison,
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
    """Failure counts, median ratio, and matrix-cluster interval."""
    interval = paired.median_speed_ratio_ci
    return {
        "n_both_solved": float(paired.n_both_solved),
        "n_reference_only": float(paired.n_reference_only),
        "n_method_only": float(paired.n_method_only),
        "speed_ratio_median": paired.median_speed_ratio,
        "speed_ratio_ci_low": interval.low if interval is not None else None,
        "speed_ratio_ci_high": interval.high if interval is not None else None,
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


@dataclass(slots=True)
class MLflowGroupReportSink:
    """Temporary event-group sink retained until method-level reporting lands."""

    manifest: FigureManifest = field(default_factory=FigureManifest)
    _staging: tempfile.TemporaryDirectory[str] = field(
        default_factory=lambda: tempfile.TemporaryDirectory(prefix="neuralls-figures-")
    )
    _finalized: bool = False

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        """Log explicit counts and the supported failure-aware plots."""
        if self._finalized:
            raise RuntimeError("cannot report after finalization")
        group_id = outcome.group_id
        log_convergence_metrics(group_id, outcome.convergence)
        log_comparison_metrics(group_id, outcome.comparisons)
        with tempfile.TemporaryDirectory() as tmp:
            file_key = sanitize_metric_key_segment(group_id)
            plans = _figure_plans(outcome, presentation, file_key)
            self.manifest.validate_plans(plans)
            outcome_plots = _write_outcome_plots(outcome, presentation, Path(tmp), file_key)
            for artifact in outcome_plots:
                plan = next(plan for plan in plans if plan.filename == artifact.name)
                mlflow.log_artifact(str(artifact), artifact_path=GROUP_REPORT_ARTIFACT_DIR)
                self.manifest.add_plan(plan)

    def finalize(self) -> None:
        """Upload exactly one session manifest while the parent run is active."""
        if self._finalized:
            raise RuntimeError("figure manifest already finalized")
        path = Path(self._staging.name) / "figure_manifest.toml"
        write_figure_manifest(path, self.manifest)
        mlflow.log_artifact(str(path), artifact_path=GROUP_REPORT_ARTIFACT_DIR)
        self._finalized = True
        self._staging.cleanup()


def _figure_kind(filename: str) -> FigureKind:
    stem = Path(filename).name
    for kind in FigureKind:
        if stem.startswith(kind.value + "_"):
            return kind
    raise ValueError(f"unknown figure kind: {filename}")


def _figure_plans(
    outcome: GroupAggregated,
    presentation: GroupPresentation,
    file_key: str,
) -> tuple[FigurePlan, ...]:
    """Build and reserve every figure's scientific context before drawing."""
    names = _planned_figure_names(outcome, file_key)
    plans: list[FigurePlan] = []
    for name in names:
        plan = make_plan(
            _figure_kind(name),
            _figure_metadata(_figure_kind(name), outcome, presentation, name),
            group_slug=file_key,
        )
        plans.append(replace(plan, filename=name))
    return tuple(plans)


def _planned_figure_names(outcome: GroupAggregated, file_key: str) -> tuple[str, ...]:
    if isinstance(outcome, RecipeProfileReport):
        return (
            f"aggregate_bars_{file_key}.png",
            f"solve_time_bars_{file_key}.png",
            f"online_time_bars_{file_key}.png",
            f"paired_time_scatter_{file_key}.png",
            f"speed_ratio_cdf_{file_key}.png",
            f"performance_profile_solve_{file_key}.png",
            f"performance_profile_online_{file_key}.png",
            f"ecdf_iterations_{file_key}.png",
            f"ecdf_time_solve_{file_key}.png",
            f"ecdf_time_online_{file_key}.png",
        )
    if isinstance(outcome, WorkloadMixtureReport):
        return (
            f"aggregate_bars_{file_key}.png",
            f"solve_time_bars_{file_key}.png",
            f"online_time_bars_{file_key}.png",
        )
    return (
        f"solve_time_bars_{file_key}.png",
        f"online_time_bars_{file_key}.png",
        f"paired_time_scatter_{file_key}.png",
        f"speed_ratio_cdf_{file_key}.png",
        f"performance_profile_solve_{file_key}.png",
        f"performance_profile_online_{file_key}.png",
        f"ecdf_iterations_{file_key}.png",
        f"ecdf_time_solve_{file_key}.png",
        f"ecdf_time_online_{file_key}.png",
    )


def _figure_metadata(
    kind: FigureKind,
    outcome: GroupAggregated,
    presentation: GroupPresentation,
    filename: str,
) -> FigureMetadata:
    scenario = (
        TimingScenario.SOLVE
        if "solve" in filename
        else TimingScenario.ONLINE
        if kind
        in {
            FigureKind.ONLINE_TIME,
            FigureKind.PAIRED_TIME,
            FigureKind.ECDF_TIME,
            FigureKind.PERFORMANCE_PROFILE,
        }
        else None
    )
    report_level = (
        "fine"
        if isinstance(outcome, RecipeProfileReport)
        else "medium"
        if isinstance(outcome.label, MediumLabel)
        else "global"
    )
    datasets = () if isinstance(outcome.label, GlobalLabel) else (str(outcome.label.dataset),)
    recipes = outcome.recipe_membership
    weights = tuple(1.0 / len(recipes) for _ in recipes) if recipes else ()
    reference = outcome.reference_selection
    reason = reference.reason if isinstance(reference, ReferenceUnavailable) else None
    methods = tuple(
        sorted(
            {str(name) for name in outcome.outcomes}
            | {str(summary.method) for summary in outcome.summaries}
        )
    )
    sources = tuple(
        source
        for method in methods
        for source in (
            f"comparison_method[method={method}]/method_workload/setup_events.parquet",
            f"comparison_method[method={method}]/method_workload/solve_attempts.parquet",
            f"comparison_method[method={method}]/method_workload/summary.toml",
            f"comparison_method[method={method}]/method_workload/method_diagnostics.json",
            f"comparison_method[method={method}]/method_workload/figures/residual_history.png",
            f"comparison_method[method={method}]/method_workload/figures/energy_history.png",
        )
    )
    metric = {
        FigureKind.AGGREGATE_BARS: "converged systems",
        FigureKind.SOLVE_TIME: "solve seconds",
        FigureKind.ONLINE_TIME: "amortized online seconds",
        FigureKind.PAIRED_TIME: "reference/method amortized online seconds",
        FigureKind.PERFORMANCE_PROFILE: "performance ratio",
        FigureKind.ECDF_ITERATIONS: "iterations",
        FigureKind.ECDF_TIME: "time seconds",
        FigureKind.SPEED_RATIO: "reference/method speed ratio",
    }.get(kind, kind.value)
    return FigureMetadata(
        kind=kind,
        metric_definition=metric,
        scenario=scenario,
        report_level=report_level,
        scope=presentation.subject,
        datasets=datasets,
        recipes=recipes,
        n_matrices=outcome.n_matrices,
        n_planned_rhs=outcome.n_planned_rhs,
        reference_selection=reference,
        failure_treatment=FailureTreatment.CENSORED,
        timing_policy="setup once per dataset/matrix + solve per planned RHS",
        recipe_weights=weights,
        reference_reason=reason,
        sources=sources,
    )


METHOD_ARTIFACT_DIR = "method_workload"
BENCHMARK_METHOD_SCHEMA_VERSION = BENCHMARK_EVENT_SCHEMA_VERSION


class MLflowBenchmarkMethodSink:
    """Write one reduced method workload into the already-active child run.

    Run ownership deliberately stays in ``comparison_batch``.  Calling this
    sink while no run is active is a caller error, just like calling
    ``mlflow.log_metric`` directly; the sink never silently creates a nested
    system or preconditioner run.
    """

    def record_method(
        self,
        workload: MethodWorkload,
        summary: MethodWorkloadSummary,
        *,
        root: Path | None = None,
    ) -> None:
        """Persist canonical event artifacts and one copy of each selected scalar."""
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            temporary = tempfile.TemporaryDirectory(prefix="neuralls-method-")
            root = Path(temporary.name)
        try:
            artifacts = write_method_benchmark_artifacts(workload, summary, root)
            self._log_method_metadata(workload, summary)
            for relative in (artifacts.setup_events, artifacts.solve_attempts, artifacts.summary):
                mlflow.log_artifact(str(root / relative), artifact_path=METHOD_ARTIFACT_DIR)
            config_path = _write_method_config(root, workload.method_config)
            mlflow.log_artifact(str(config_path), artifact_path=METHOD_ARTIFACT_DIR)
            for diagnostics_path in _write_method_diagnostics(root, workload):
                artifact_path = METHOD_ARTIFACT_DIR
                if diagnostics_path.parent.name == "figures":
                    artifact_path = f"{METHOD_ARTIFACT_DIR}/figures"
                mlflow.log_artifact(str(diagnostics_path), artifact_path=artifact_path)
        finally:
            if temporary is not None:
                temporary.cleanup()

    @staticmethod
    def _log_method_metadata(workload: MethodWorkload, summary: MethodWorkloadSummary) -> None:
        """Log only reducer-owned searchable scalars, never event-loop aggregates."""
        mlflow.log_param("benchmark_event_schema_version", str(BENCHMARK_METHOD_SCHEMA_VERSION))
        mlflow.log_param("method", str(workload.method))
        mlflow.log_param("method_config_digest", str(workload.config_digest))
        metrics = {
            "n_planned_rhs": float(summary.n_planned_rhs),
            "n_attempted_rhs": float(summary.n_attempted_rhs),
            "n_converged": float(summary.n_converged),
            "success_rate": (
                summary.n_converged / summary.n_planned_rhs if summary.n_planned_rhs else 0.0
            ),
            "setup_total_s": summary.setup_total_s,
            "solve_total_s": summary.solve_total_s,
            "workload_online_s": summary.workload_online_s,
        }
        if summary.online_s_per_attempted_rhs is not None:
            metrics["online_s_per_attempted_rhs"] = summary.online_s_per_attempted_rhs
        from neuralls.domain.solver.models.benchmark_events import MeasuredPeakMemory

        if isinstance(summary.workload_peak_memory, MeasuredPeakMemory):
            metrics["workload_peak_memory_bytes"] = float(summary.workload_peak_memory.bytes)
        if workload.generation_cost is not None:
            metrics["generation_s"] = workload.generation_cost.wall_time_seconds
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


def _write_method_config(root: Path, config: Mapping[str, object]) -> Path:
    """Stage a method config without making config serialization part of timing."""
    path = root / "method_config.toml"
    import tomli_w

    path.write_bytes(tomli_w.dumps(_exclude_none(config)).encode("utf-8"))
    return path


def _exclude_none(config: Mapping[str, object]) -> dict[str, object]:
    """Recursively remove unset model fields before TOML serialization."""
    return {key: _exclude_none_value(value) for key, value in config.items() if value is not None}


def _exclude_none_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _exclude_none(value)
    if isinstance(value, list):
        return [_exclude_none_value(item) for item in value]
    return value


def _write_method_diagnostics(root: Path, workload: MethodWorkload) -> tuple[Path, ...]:
    """Stage histories and method figures without creating child runs."""
    diagnostics = {str(system_id): result for system_id, result in workload.diagnostics}
    path = root / "method_diagnostics.json"
    payload = [
        {
            "system_id": str(system_id),
            "method": str(workload.method),
            "iterations": result.iterations,
            "converged": result.converged,
            "solve_s": result.solve_time_seconds,
            "residual_history_rel": list(result.residual_history_rel),
            "error_history_a_rel": (
                list(result.error_history_a_rel) if result.error_history_a_rel is not None else None
            ),
        }
        for system_id, result in workload.diagnostics
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    residuals_path = root / "residual_histories.json"
    residuals_path.write_text(
        json.dumps(
            {name: list(result.residual_history_rel) for name, result in diagnostics.items()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    energies_path = root / "energy_histories.json"
    energies_path.write_text(
        json.dumps(
            {
                name: (
                    list(result.error_history_a_rel)
                    if result.error_history_a_rel is not None
                    else None
                )
                for name, result in diagnostics.items()
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    residual_figure = figures / "residual_history.png"
    plot_convergence_comparison(diagnostics, save_path=residual_figure, title=str(workload.method))
    energy_figure = figures / "energy_history.png"
    plot_error_convergence_comparison(
        diagnostics, save_path=energy_figure, title=str(workload.method)
    )
    return path, residuals_path, energies_path, residual_figure, energy_figure


def _group_style(
    kind: str, presentation: GroupPresentation, plot_style: PlotStyleMaps
) -> GroupStyle:
    """The shared concise title and per-preconditioner styling of one plot."""
    titles = {
        "Performance profile (time)": "Performance Profile",
        "Performance profile (solve)": "Performance Profile",
        "Performance profile (online)": "Performance Profile",
        "ECDF (iterations)": "Iterations",
        "ECDF (time)": "Online Time",
        "ECDF (solve time)": "Solve Time",
        "ECDF (online time)": "Online Time",
        "Time to solution vs reference": "Online Time",
        "Solve time": "Solve Time",
        "Online time": "Online Time",
        "Reference/method speed ratio": "Speed Ratio",
        "Aggregate convergence": "Convergence",
    }
    return GroupStyle(
        title=titles.get(kind, kind),
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
    """Draw fine-only iteration and separate solve/online time profiles."""
    plot_style = plot_style_maps(presentation.plot_styles)
    return [
        plot_performance_profile(
            outcome.outcomes,
            style=_group_style("Performance profile (solve)", presentation, plot_style),
            cost=solve_seconds,
            save_path=directory / f"performance_profile_solve_{file_key}.png",
        ),
        plot_performance_profile(
            outcome.outcomes,
            style=_group_style("Performance profile (online)", presentation, plot_style),
            cost=amortized_online_seconds,
            save_path=directory / f"performance_profile_online_{file_key}.png",
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=_group_style("ECDF (iterations)", presentation, plot_style),
            save_path=directory / f"ecdf_iterations_{file_key}.png",
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=_group_style("ECDF (solve time)", presentation, plot_style),
            cost=solve_seconds,
            xlabel="Solve time per system [s]",
            save_path=directory / f"ecdf_time_solve_{file_key}.png",
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=_group_style("ECDF (online time)", presentation, plot_style),
            cost=amortized_online_seconds,
            xlabel="Amortized online time per system [s]",
            save_path=directory / f"ecdf_time_online_{file_key}.png",
        ),
    ]


def _write_system_plots(
    outcome: GroupAggregated, presentation: GroupPresentation, directory: Path, file_key: str
) -> list[Path | None]:
    """Draw setup-aware timing bars, paired timing, and ratio plots."""
    plot_style = plot_style_maps(presentation.plot_styles)
    paired = (
        plot_paired_time_scatter(
            outcome.outcomes,
            reference=outcome.reference_selection,
            style=_group_style("Time to solution vs reference", presentation, plot_style),
            save_path=directory / f"paired_time_scatter_{file_key}.png",
        )
        if isinstance(outcome.reference_selection, ConfiguredReference)
        else None
    )
    plots: list[Path | None] = [
        plot_online_time_bars(
            outcome.summaries,
            scenario="solve",
            style=_group_style("Solve time", presentation, plot_style),
            save_path=directory / f"solve_time_bars_{file_key}.png",
        ),
        plot_online_time_bars(
            outcome.summaries,
            scenario="online",
            style=_group_style("Online time", presentation, plot_style),
            save_path=directory / f"online_time_bars_{file_key}.png",
        ),
        paired,
        plot_speed_ratio_cdf(
            _paired_by_method(outcome.comparisons),
            style=_group_style("Reference/method speed ratio", presentation, plot_style),
            reference_name=outcome.reference_selection.name
            if isinstance(outcome.reference_selection, ConfiguredReference)
            else None,
            save_path=directory / f"speed_ratio_cdf_{file_key}.png",
        ),
    ]
    return plots


def _write_outcome_plots(
    outcome: GroupAggregated | RecipeProfileReport | WorkloadMixtureReport,
    presentation: GroupPresentation,
    directory: Path,
    file_key: str,
) -> tuple[Path, ...]:
    """Draw plots allowed by the typed report variant."""
    from typing import assert_never

    match outcome:
        case RecipeProfileReport():
            aggregate_path = directory / f"aggregate_bars_{file_key}.png"
            aggregate = plot_aggregate_bars(
                outcome,
                style=_group_style(
                    "Aggregate convergence", presentation, plot_style_maps(presentation.plot_styles)
                ),
                save_path=aggregate_path,
            )
            drawn = [
                aggregate,
                *_write_system_plots(outcome, presentation, directory, file_key),
                *_write_profile_plots(outcome, presentation, directory, file_key),
            ]
        case WorkloadMixtureReport():
            aggregate_path = directory / f"aggregate_bars_{file_key}.png"
            aggregate = plot_aggregate_bars(
                outcome,
                style=_group_style(
                    "Aggregate convergence", presentation, plot_style_maps(presentation.plot_styles)
                ),
                save_path=aggregate_path,
            )
            drawn = [
                aggregate,
                plot_online_time_bars(
                    outcome.summaries,
                    scenario="solve",
                    style=_group_style(
                        "Solve time", presentation, plot_style_maps(presentation.plot_styles)
                    ),
                    save_path=directory / f"solve_time_bars_{file_key}.png",
                ),
                plot_online_time_bars(
                    outcome.summaries,
                    scenario="online",
                    style=_group_style(
                        "Online time", presentation, plot_style_maps(presentation.plot_styles)
                    ),
                    save_path=directory / f"online_time_bars_{file_key}.png",
                ),
            ]
        case GroupAggregated():
            drawn = _write_system_plots(outcome, presentation, directory, file_key)
            drawn.extend(_write_profile_plots(outcome, presentation, directory, file_key))
        case _ as unreachable:
            assert_never(unreachable)
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
