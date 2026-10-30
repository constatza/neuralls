"""Run discovery for a comparison session: its method children and their stored artifacts.

The one place that knows how method runs are tagged and where their artifacts live, so the
compare reuse gate and the report workflow cannot disagree about either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from neuralls.platform.reporting.benchmark_events import (
    METHOD_ARTIFACT_DIR,
    MethodBenchmarkArtifacts,
)
from neuralls.platform.reporting.stored_benchmark import MethodDirectory
from neuralls.platform.tracking.mlflow import quote_filter_value
from neuralls.shared.types import SessionRunId

METHOD_PHASE: Final = "comparison_method"
REPORT_PHASE: Final = "report"
PARENT_RUN_TAG = "mlflow.parentRunId"
REQUIRED_METHOD_ARTIFACTS = frozenset(MethodBenchmarkArtifacts.file_names())
_NEWEST_FIRST = ["attributes.start_time DESC", "attributes.run_id ASC"]


class MethodArtifactsMissingError(ValueError):
    """A method run lacks part of its required artifact set."""

    def __init__(self, run_id: str, missing: list[str]) -> None:
        self.run_id = run_id
        self.missing = missing
        super().__init__(f"method run {run_id} is missing artifacts {missing}")


def method_artifact_paths(client: MlflowClient, run_id: str) -> dict[str, str]:
    """Artifact file name -> artifact path, for the files a method run logged."""
    return {
        info.path.rsplit("/", 1)[-1]: info.path
        for info in client.list_artifacts(run_id, path=METHOD_ARTIFACT_DIR)
        if not info.is_dir
    }


def session_method_runs(client: MlflowClient, session_run_id: SessionRunId) -> list[Run]:
    """The FINISHED method children of a session, oldest first."""
    experiment_id = client.get_run(session_run_id).info.experiment_id
    clauses = [
        "attributes.status = 'FINISHED'",
        f"tags.`phase` = '{METHOD_PHASE}'",
        f"tags.`{PARENT_RUN_TAG}` = '{quote_filter_value(session_run_id)}'",
    ]
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=" and ".join(clauses),
        order_by=_NEWEST_FIRST,
    )
    return list(reversed(runs))


def download_session_methods(
    client: MlflowClient, session_run_id: SessionRunId, destination: Path
) -> list[MethodDirectory]:
    """Download every method child's artifacts; return each local directory with its run id.

    Raises:
        ValueError: The session has no FINISHED method child.
        MethodArtifactsMissingError: A method child lacks part of the required artifact set.
    """
    runs = session_method_runs(client, session_run_id)
    if not runs:
        raise ValueError(f"session {session_run_id} has no finished method runs")
    directories: list[MethodDirectory] = []
    for run in runs:
        run_id = run.info.run_id
        paths = method_artifact_paths(client, run_id)
        missing = sorted(REQUIRED_METHOD_ARTIFACTS - paths.keys())
        if missing:
            raise MethodArtifactsMissingError(run_id, missing)
        local = client.download_artifacts(
            run_id, path=METHOD_ARTIFACT_DIR, dst_path=str(destination / run_id)
        )
        directories.append(
            MethodDirectory(run_id, Path(local), run.data.tags["method_config_digest"])
        )
    return directories


SESSION_PHASE: Final = "session_comparison"


def latest_session_run_id(
    client: MlflowClient, experiment_name: str, case_config_path: str
) -> SessionRunId:
    """The newest FINISHED comparison session of a case config.

    Raises:
        ValueError: The experiment or a finished session for the config does not exist.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"experiment {experiment_name!r} does not exist; run `neuralls compare`")
    clauses = [
        "attributes.status = 'FINISHED'",
        f"tags.`phase` = '{SESSION_PHASE}'",
        f"tags.`case_config_path` = '{quote_filter_value(case_config_path)}'",
    ]
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=" and ".join(clauses),
        order_by=_NEWEST_FIRST,
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"no finished comparison session for {case_config_path}; run `neuralls compare`"
        )
    return SessionRunId(runs[0].info.run_id)
