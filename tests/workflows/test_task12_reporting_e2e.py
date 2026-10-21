"""Deterministic end-to-end acceptance coverage for setup-aware reporting."""

from __future__ import annotations

import tomllib
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient

from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.models import CompletedMethodWorkload, canonical_method_config
from neuralls.composition.tracking.run_specs import build_benchmark_method_run_spec
from neuralls.domain.benchmarking.selection import ConfiguredReference
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.models.workload import CompleteWorkloadSummary, MethodWorkloadSummary
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.classification import classify_level
from neuralls.domain.solver.reporting.event_frames import SETUP_EVENT_SCHEMA, SOLVE_ATTEMPT_SCHEMA
from neuralls.domain.solver.reporting.report_levels import FineLevel
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting import outcome_plots
from neuralls.platform.reporting.benchmark_events import read_setup_events, read_solve_attempts
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    METHOD_ARTIFACT_DIR,
    MLflowBenchmarkMethodSink,
    MLflowGroupReportSink,
)
from neuralls.shared.digest import canonical_digest
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId

DATASET = DatasetId("sinusoidal-workload")
RECIPES = ("sinusoidal ω=1", "sinusoidal ω=2")
MATRIX_IDS = (MatrixId(5), MatrixId(9))
METHOD_METRICS = {
    "n_planned_rhs",
    "n_attempted_rhs",
    "n_converged",
    "success_rate",
    "setup_total_s",
    "solve_total_s",
    "workload_online_s",
    "online_s_per_attempted_rhs",
    "workload_peak_memory_bytes",
}


def _tags() -> tuple[SystemTag, ...]:
    return tuple(
        SystemTag(
            DATASET,
            recipe,
            matrix_id,
            rhs_ordinal,
            SystemId(f"m{matrix_id}-rhs{rhs_ordinal}"),
        )
        for matrix_id in MATRIX_IDS
        for rhs_ordinal, recipe in enumerate(RECIPES)
    )


def _diagnostic(
    method: PreconditionerName, *, converged: bool, iterations: int
) -> CGComparisonResult:
    residual = 1.0e-8 if converged else 1.0
    return CGComparisonResult(
        x=np.zeros(2),
        converged=converged,
        iterations=iterations,
        residual=residual,
        residual_abs=residual,
        residual_history_rel=[1.0, residual],
        residual_history_abs=[1.0, residual],
        preconditioner=str(method),
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=not converged,
        error=None if converged else "deterministic identity iteration limit",
        solve_time_seconds=0.0,
    )


def _workload(
    method: PreconditionerName,
    config: StandardPreconditionerConfig,
    setup_seconds: tuple[float, float],
    solve_seconds: tuple[float, float, float, float],
    *,
    converged: bool,
) -> CompletedMethodWorkload:
    tags = _tags()
    setup_events = tuple(
        MatrixSetupEvent(
            event_id=SetupEventId(f"{method}-setup-{matrix_id}"),
            dataset=DATASET,
            matrix_id=matrix_id,
            method=method,
            setup_s=seconds,
            samples_s=(seconds / 2, seconds, seconds * 1.5),
            peak_memory=MeasuredPeakMemory(8 if method == "identity" else 16),
            outcome=SetupCompleted(),
        )
        for matrix_id, seconds in zip(MATRIX_IDS, setup_seconds, strict=True)
    )
    attempts = tuple(
        SolveAttempt(
            setup_event_id=SetupEventId(f"{method}-setup-{tag.matrix_id}"),
            system_id=tag.system_id,
            dataset=tag.dataset,
            recipe=tag.recipe,
            matrix_id=tag.matrix_id,
            rhs_ordinal=tag.rhs_ordinal,
            method=method,
            solve_s=seconds,
            samples_s=(seconds / 2, seconds, seconds * 1.5),
            peak_memory=MeasuredPeakMemory(32 if method == "identity" else 64),
            outcome=(
                SolveConverged(4 + index, 1.0e-8)
                if converged
                else SolveFailed(20, "deterministic identity iteration limit")
            ),
        )
        for index, (tag, seconds) in enumerate(zip(tags, solve_seconds, strict=True))
    )
    method_config = canonical_method_config(config)
    return CompletedMethodWorkload(
        method=method,
        planned=tags,
        setup_events=setup_events,
        solve_attempts=attempts,
        diagnostics=tuple(
            (tag.system_id, _diagnostic(method, converged=converged, iterations=4 + index))
            for index, tag in enumerate(tags)
        ),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(128 if method == "identity" else 256),
        config_digest=canonical_digest(method_config),
        method_config=method_config,
    )


@pytest.fixture
def e2e_workloads() -> tuple[CompletedMethodWorkload, ...]:
    """Two physical matrices, two sinusoidal RHS frequencies, and two methods."""
    identity = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    return (
        _workload(
            PreconditionerName("identity"),
            identity,
            (0.0, 0.0),
            (0.20, 0.20, 0.20, 0.20),
            converged=False,
        ),
        _workload(
            PreconditionerName("jacobi"),
            jacobi,
            (0.25, 0.45),
            (0.20, 0.30, 0.25, 0.35),
            converged=True,
        ),
    )


def _artifact_names(client: MlflowClient, run_id: str, path: str) -> set[str]:
    return {Path(item.path).name for item in client.list_artifacts(run_id, path=path)}


def _all_runs(client: MlflowClient, experiment_name: str):
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    return client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time ASC"]
    )


@pytest.mark.integration
@pytest.mark.real_render
def test_setup_aware_reporting_reconciles_events_figures_and_mlflow(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    e2e_workloads: tuple[CompletedMethodWorkload, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist one deterministic workload and reconcile every reporting boundary."""
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    captured: list[tuple[Figure, str]] = []
    finish = outcome_plots.finish_group_figure

    def capture_figure(fig: Figure, ax: Axes, *, title: str, **kwargs: object) -> None:
        captured.append((fig, title))
        finish(fig, ax, title=title, **kwargs)

    monkeypatch.setattr(outcome_plots, "finish_group_figure", capture_figure)
    method_roots: dict[PreconditionerName, Path] = {}
    summaries: dict[PreconditionerName, MethodWorkloadSummary] = {}

    with mlflow.start_run(run_name="task12-session") as session:
        session_id = session.info.run_id
        method_sink = MLflowBenchmarkMethodSink()
        for workload in e2e_workloads:
            method_root = tmp_path / str(workload.method)
            method_roots[workload.method] = method_root
            summary = summarize_method_workload(
                workload.method,
                workload.planned,
                workload.setup_events,
                workload.solve_attempts,
                workload.workload_peak_memory,
            )
            summaries[workload.method] = summary
            assert isinstance(summary, CompleteWorkloadSummary)
            with mlflow.start_run(
                run_name=f"method | {workload.method}",
                nested=True,
                tags=build_benchmark_method_run_spec(
                    method=str(workload.method),
                    config_digest=workload.config_digest,
                    system_ids=tuple(str(tag.system_id) for tag in workload.planned),
                )[1].as_mlflow_tags(),
            ):
                method_sink.record_method(workload, summary, root=method_root)

        report_sink = MLflowGroupReportSink()
        report_all_levels(
            e2e_workloads,
            report_sink,
            reference_selection=ConfiguredReference(PreconditionerName("jacobi")),
        )
        report_sink.finalize()

    # The persisted event artifacts are canonical and retain physical matrix ids, links, samples,
    # and the measured zero-cost setup rather than manufacturing setup rows per RHS.
    for workload in e2e_workloads:
        root = method_roots[workload.method]
        setup = read_setup_events(root / "setup_events.parquet")
        solves = read_solve_attempts(
            root / "solve_attempts.parquet", setup_events_path=root / "setup_events.parquet"
        )
        assert setup.schema == pl.Schema(SETUP_EVENT_SCHEMA)
        assert solves.schema == pl.Schema(SOLVE_ATTEMPT_SCHEMA)
        assert setup.height == 2
        assert solves.height == 4
        assert set(setup["matrix_id"].to_list()) == {5, 9}
        assert set(solves["matrix_id"].to_list()) == {5, 9}
        assert set(solves["setup_event_id"].to_list()) == set(setup["setup_event_id"].to_list())
        assert all(len(samples) == 3 for samples in setup["samples_s"].to_list())
        assert all(len(samples) == 3 for samples in solves["samples_s"].to_list())
        assert setup.filter(pl.col("method") == str(workload.method)).height == 2
        expected_setup = (0.0, 0.0) if workload.method == "identity" else (0.25, 0.45)
        assert setup["setup_s"].to_list() == list(expected_setup)
        assert tomllib.loads((root / "summary.toml").read_text(encoding="utf-8"))["counts"] == {
            "planned_rhs": 4,
            "attempted_rhs": 4,
            "converged": summaries[workload.method].n_converged,
        }
        summary_payload = tomllib.loads((root / "summary.toml").read_text(encoding="utf-8"))
        online = summary_payload["online"]
        assert online["setup_s"] == pytest.approx(setup["setup_s"].sum())
        assert online["solve_s"] == pytest.approx(solves["solve_s"].sum())
        assert online["workload_s"] == pytest.approx(online["setup_s"] + online["solve_s"])
        assert online["per_attempted_rhs_s"] == pytest.approx(online["workload_s"] / 4)

    assert summaries[PreconditionerName("identity")].setup_total_s == 0.0
    assert summaries[PreconditionerName("identity")].n_converged == 0
    assert summaries[PreconditionerName("jacobi")].setup_total_s == pytest.approx(0.70)
    assert summaries[PreconditionerName("jacobi")].n_converged == 4
    fine_reports = classify_level(
        _tags(),
        tuple(event for workload in e2e_workloads for event in workload.setup_events),
        tuple(attempt for workload in e2e_workloads for attempt in workload.solve_attempts),
        FineLevel(),
        ConfiguredReference(PreconditionerName("jacobi")),
    )
    assert len(fine_reports) == 2
    for report in fine_reports:
        assert isinstance(report, RecipeProfileReport)
        assert report.recipe_membership == (report.label.recipe,)
        assert report.n_matrices == 2 and report.n_planned_rhs == 2
        asymmetric = report.comparisons[PreconditionerName("identity")].paired
        assert (
            asymmetric.n_both_solved,
            asymmetric.n_reference_only,
            asymmetric.n_method_only,
        ) == (
            0,
            2,
            0,
        )

    runs = _all_runs(client, mlflow_experiment)
    assert len(runs) == 3
    roots = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    assert len(roots) == 1 and roots[0].info.run_id == session_id
    assert not {"setup_total_s", "solve_total_s", "workload_online_s"}.intersection(
        roots[0].data.metrics
    )
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == session_id]
    assert {run.data.tags["phase"] for run in children} == {"comparison_method"}
    assert {run.data.params["method"] for run in children} == {"identity", "jacobi"}
    assert not any(
        run.data.tags.get("mlflow.parentRunId") in {child.info.run_id for child in children}
        for run in runs
    )
    for child in children:
        assert set(child.data.metrics) == METHOD_METRICS
        assert _artifact_names(client, child.info.run_id, METHOD_ARTIFACT_DIR) >= {
            "setup_events.parquet",
            "solve_attempts.parquet",
            "summary.toml",
            "method_config.toml",
            "method_diagnostics.json",
        }

    report_names = _artifact_names(client, session_id, GROUP_REPORT_ARTIFACT_DIR)
    manifest_path = client.download_artifacts(
        session_id, f"{GROUP_REPORT_ARTIFACT_DIR}/figure_manifest.toml", str(tmp_path)
    )
    manifest = tomllib.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_names = set(manifest["figures"])
    assert report_names == manifest_names | {"figure_manifest.toml"}
    assert len([name for name in manifest_names if name.startswith("performance_profile_")]) == 2
    assert len([name for name in manifest_names if name.startswith("ecdf_")]) == 4
    assert len([name for name in manifest_names if name.startswith("aggregate_bars_")]) == 4
    assert (
        len(
            [
                name
                for name in manifest_names
                if name.startswith(("aggregate_bars_medium_", "aggregate_bars_global"))
            ]
        )
        == 2
    )
    for name, metadata in manifest["figures"].items():
        assert metadata["sources"] == [
            "setup_events.parquet",
            "solve_attempts.parquet",
            "summary.toml",
        ]
        assert metadata["reference_selection"] == "configured:jacobi"
        if name.startswith(("performance_profile_", "ecdf_")):
            assert metadata["report_level"] == "fine"
        if name.startswith(("aggregate_bars_medium_", "aggregate_bars_global_")):
            assert metadata["report_level"] == "mixture"
    assert all(len(title) <= 24 for _, title in captured)
    convergence_figures = [fig for fig, title in captured if title == "Convergence"]
    assert len(convergence_figures) == 4
    assert all(fig.axes[0].patches for fig in convergence_figures)
    paired_figures = [
        fig
        for fig, _ in captured
        if "(reference): amortized online time" in fig.axes[0].get_xlabel()
    ]
    assert paired_figures
    assert all(
        fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == "linear" for fig in paired_figures
    )
    assert not any("speedup" in title.lower() or "ert" in title.lower() for _, title in captured)
