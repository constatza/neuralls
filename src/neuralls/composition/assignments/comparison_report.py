"""Report workflow: read a finished compare session back and log its figures and metrics.

Never solves or times. The session is read from the comparison experiment and everything is
written to the report experiment, so a failure here cannot invalidate completed compute.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
from loguru import logger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.assignment_subjects import case_notation
from neuralls.composition.assignments.comparison_batch import (
    _failure_boundary,
    _load_master_config,
    _resolve_comparison_topology,
)
from neuralls.composition.comparison.postprocessing.group_reporting import report_all_levels
from neuralls.composition.comparison.postprocessing.ports import ReportSink
from neuralls.composition.comparison.postprocessing.report_inputs import (
    ReportInputs,
    report_inputs,
)
from neuralls.composition.comparison.postprocessing.system_figures import write_system_figures
from neuralls.domain.solver.models.benchmark_events import SolveConverged
from neuralls.domain.solver.models.report_scope import AllMatrixSets, System
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.figure_index import FIGURE_INDEX_FILE, read_figure_index
from neuralls.platform.reporting.stored_benchmark import read_stored_benchmark
from neuralls.platform.tracking.comparison_tracking import (
    MLflowReportSink,
    parent_run_metric_key,
    setup_comparison_tracking,
)
from neuralls.platform.tracking.failure_reporting import MLflowWorkflowFailureSink
from neuralls.platform.tracking.mlflow import quote_filter_value
from neuralls.platform.tracking.scope_params import scope_params
from neuralls.platform.tracking.session_artifacts import (
    REPORT_PHASE,
    download_session_methods,
    latest_session_run_id,
)
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import SessionRunId, SystemId
from neuralls.shared.workflow_failures import ComparisonBatchFailureContext, ComparisonBatchPhase

REPORT_SCHEMA_VERSION = "2"


@dataclass(frozen=True, slots=True)
class ComparisonReportOutcome:
    """The report run written for one comparison session, and whether it was already there."""

    report_run_id: str
    session_run_id: SessionRunId
    reused: bool


def run_comparison_report(
    case_config_path: Path,
    *,
    force: bool = False,
    session_run_id: SessionRunId | None = None,
    settings: NeurallsSettings | None = None,
) -> ComparisonReportOutcome:
    """Render the newest (or the named) comparison session of a case config into a report run."""
    settings = require_settings(settings, case_config_path=case_config_path)
    master_cfg, _ = _load_master_config(case_config_path, settings)
    topology = _resolve_comparison_topology(case_config_path, settings)
    notation = case_notation(master_cfg)
    client = MlflowClient(tracking_uri=topology.tracking_uri)
    session = session_run_id or latest_session_run_id(
        client, topology.experiment_name, case_config_path.resolve().as_posix()
    )
    notation_digest = hashlib.sha256(repr(notation).encode("utf-8")).hexdigest()[:8]
    source_tags = {
        "source_session_run_id": session,
        "notation_digest": notation_digest,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }

    report_experiment = topology.report_experiment_name
    setup_comparison_tracking(
        tracking_uri=topology.tracking_uri,
        artifact_location=topology.artifact_location,
        experiment_name=report_experiment,
    )
    existing = _finished_report(client, report_experiment, source_tags)
    if existing is not None and not force:
        logger.info("Report {} already covers session {}.", existing, session)
        return ComparisonReportOutcome(existing, session, reused=True)

    with tempfile.TemporaryDirectory(prefix="neuralls-report-") as tmp:
        directories = download_session_methods(client, session, Path(tmp))
        inputs = report_inputs(read_stored_benchmark(directories))
        session_name = client.get_run(session).data.tags.get("mlflow.runName", session)
        failure_sink = MLflowWorkflowFailureSink()
        with mlflow.start_run(
            run_name=session_name,
            tags={
                "phase": REPORT_PHASE,
                "report_level": "all_matrix_sets",
                "source_experiment": topology.experiment_name,
                **source_tags,
            },
        ) as run:
            mlflow.log_params(scope_params(AllMatrixSets()))
            sink = MLflowReportSink(
                notation, session, methods=inputs.stored.methods, datasets=inputs.datasets
            )
            with _failure_boundary(
                failure_sink,
                ComparisonBatchFailureContext(
                    phase=ComparisonBatchPhase.REPORTING,
                    case_config_path=case_config_path.resolve(),
                ),
            ):
                sink.register_method_figures(datasets=inputs.datasets)
                report_all_levels(
                    inputs.methods,
                    sink,
                    datasets=inputs.datasets,
                    notation=notation,
                    reference_selection=dict(inputs.references),
                    contexts_by_system=inputs.solver_contexts,
                )
                _report_systems(sink, inputs, notation)
            with _failure_boundary(
                failure_sink,
                ComparisonBatchFailureContext(
                    phase=ComparisonBatchPhase.FINALIZATION,
                    case_config_path=case_config_path.resolve(),
                ),
            ):
                sink.finalize()
    return ComparisonReportOutcome(run.info.run_id, session, reused=False)


def _finished_report(
    client: MlflowClient, experiment_name: str, tags: dict[str, str]
) -> str | None:
    """The newest FINISHED report run carrying exactly these source tags and a valid figure index."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    clauses = ["attributes.status = 'FINISHED'", f"tags.`phase` = '{REPORT_PHASE}'"] + [
        f"tags.`{key}` = '{quote_filter_value(value)}'" for key, value in tags.items()
    ]
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=" and ".join(clauses),
        order_by=["attributes.start_time DESC"],
    )
    for run in runs:
        if _has_valid_figure_index(client, run.info.run_id):
            return run.info.run_id
        logger.info("Report {} has no valid {}; not reusable.", run.info.run_id, FIGURE_INDEX_FILE)
    return None


def _has_valid_figure_index(client: MlflowClient, run_id: str) -> bool:
    """Whether the run's ``figure_index.toml`` exists and parses as a complete index."""
    with tempfile.TemporaryDirectory(prefix="neuralls-report-index-") as tmp:
        try:
            local = client.download_artifacts(run_id, path=FIGURE_INDEX_FILE, dst_path=tmp)
            read_figure_index(Path(local))
        except MlflowException, OSError, ValueError:
            return False
    return True


def _report_systems(sink: ReportSink, inputs: ReportInputs, notation: NotationPolicy) -> None:
    """Record every system's per-method metrics and convergence figures at its own scope."""
    methods = inputs.methods
    attempts = [attempt for method in methods for attempt in method.solve_attempts]
    metrics_by_system: dict[SystemId, dict[str, float]] = {}
    for attempt in attempts:
        metrics = metrics_by_system.setdefault(attempt.system_id, {})
        metrics[parent_run_metric_key("iterations", str(attempt.method))] = float(
            attempt.outcome.iterations
        )
        metrics[parent_run_metric_key("solve_s", str(attempt.method))] = attempt.solve_s
        metrics[parent_run_metric_key("converged", str(attempt.method))] = float(
            isinstance(attempt.outcome, SolveConverged)
        )
    with tempfile.TemporaryDirectory(prefix="neuralls-system-figures-") as tmp:
        figures = write_system_figures(
            methods,
            inputs.convergence,
            datasets=inputs.datasets,
            references=inputs.references,
            solver_contexts=inputs.solver_contexts,
            notation=notation,
            root=Path(tmp),
        )
        tags = {tag.system_id: tag for method in methods for tag in method.planned}
        for system_id, tag in tags.items():
            sink.report_system(
                System(tag.dataset, tag.recipe, tag.matrix_id, tag.rhs_ordinal, system_id),
                metrics_by_system.get(system_id, {}),
                [figure.staged for figure in figures if figure.tag.system_id == system_id],
            )
