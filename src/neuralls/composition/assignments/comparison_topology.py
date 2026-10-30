"""MLflow topology and failure recording shared by the compare and report workflows."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.shared.workflow_failures import WorkflowFailureContext, WorkflowFailureSink


@dataclass(frozen=True, slots=True)
class ComparisonTopology:
    """Shared MLflow topology for comparison execution."""

    tracking_uri: str
    artifact_location: str | None
    experiment_name: str
    model_store_tracking_uri: str
    report_experiment_name: str


def load_master_config(
    case_config_path: Path,
    settings: NeurallsSettings,
) -> tuple[CaseConfig, Path]:
    """Load the case config and its resolution root."""
    return load_validated_case_config(case_config_path, settings)


def resolve_comparison_topology(
    case_config_path: Path,
    settings: NeurallsSettings,
) -> ComparisonTopology:
    """Resolve comparison MLflow topology from the case config."""
    master_cfg, _ = load_master_config(case_config_path, settings)
    runtime = build_workflow_environment(
        tracking_uri=master_cfg.mlflow.tracking_uri,
        artifact_location=master_cfg.mlflow.artifacts_destination,
        default_output_root=settings.output_dir,
        config_path=case_config_path,
    )
    return ComparisonTopology(
        tracking_uri=runtime.tracking_uri,
        artifact_location=runtime.artifact_uri,
        experiment_name=master_cfg.names.comparison,
        model_store_tracking_uri=runtime.tracking_uri,
        report_experiment_name=master_cfg.names.report,
    )


@contextlib.contextmanager
def failure_boundary(
    sink: WorkflowFailureSink,
    context: WorkflowFailureContext,
    *,
    recorded_exception_ids: set[int] | None = None,
) -> Generator[None]:
    """Record a blocking failure while preserving its exact exception and traceback."""
    try:
        yield
    except Exception as exc:
        exc.add_note(f"Comparison workflow context: {context!r}")
        exception_id = id(exc)
        if recorded_exception_ids is not None and exception_id in recorded_exception_ids:
            raise
        if recorded_exception_ids is not None:
            recorded_exception_ids.add(exception_id)
        try:
            sink.record(context, exc)
        except Exception as diagnostic_exc:  # noqa: BLE001
            logger.warning(
                "Failure diagnostics raised while preserving '{}': {}",
                exc,
                diagnostic_exc,
            )
        raise
