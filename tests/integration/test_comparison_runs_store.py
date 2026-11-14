"""The MLflow comparison-runs adapter against a real store: runs, nesting, status and records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from tests.environment_support import SESSION_ENVIRONMENT

from neuralls.domain.presentation.methods import StandaloneMethodPresentation
from neuralls.domain.presentation.run_names import MethodRunName, NamedRun
from neuralls.platform.tracking.comparison_runs import MethodRecordContext, MlflowComparisonRuns
from neuralls.platform.tracking.failure_reporting import FAILURE_CONTEXT_ARTIFACT
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.platform.tracking.run_writer import PARENT_RUN_TAG, MlflowRunWriter
from neuralls.platform.tracking.runtime_environment import environment_params
from neuralls.platform.tracking.split_records import SPLITS_ARTIFACT, encode_split_records
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.workflow_failures import ComparisonBatchFailureContext, ComparisonBatchPhase


class _InjectedFailure(RuntimeError):
    pass


@pytest.fixture
def client(mlflow_tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=mlflow_tracking_uri)


@pytest.fixture
def runs(client: MlflowClient) -> MlflowComparisonRuns:
    experiment_id = client.create_experiment("comparison-runs")
    return MlflowComparisonRuns(
        MlflowRunWriter(client), experiment_id, MethodRecordContext({}, {}, {})
    )


@pytest.fixture
def named_run() -> NamedRun:
    return NamedRun(
        MethodRunName(StandaloneMethodPresentation(ScientificDescription("Jacobi"))),
        NotationPolicy.default(),
    )


def _all_runs(client: MlflowClient, runs: MlflowComparisonRuns) -> dict[str, Run]:
    return {run.info.run_id: run for run in client.search_runs([runs.experiment_id])}


def test_a_session_and_its_method_run_finish_nested_with_their_records(
    client: MlflowClient, runs: MlflowComparisonRuns, named_run: NamedRun, tmp_path: Path
) -> None:
    with runs.session(named_run, {"phase": "session"}) as session:
        session.record_splits(())
        session.record_environment(SESSION_ENVIRONMENT)
        with session.method_run(named_run, {"phase": "method"}) as method:
            method_run_id = method.run_id
        session_run_id = session.run_id

    stored = _all_runs(client, runs)
    session_run, method_run = stored[session_run_id], stored[method_run_id]
    assert (session_run.info.status, method_run.info.status) == ("FINISHED", "FINISHED")
    assert (session_run.info.run_name, method_run.info.run_name) == (
        run_name_text(named_run),
        run_name_text(named_run),
    )
    assert PARENT_RUN_TAG not in session_run.data.tags
    assert method_run.data.tags[PARENT_RUN_TAG] == session_run_id
    assert (session_run.data.tags["phase"], method_run.data.tags["phase"]) == (
        "session",
        "method",
    )
    assert session_run.data.params == environment_params(SESSION_ENVIRONMENT)
    splits = Path(client.download_artifacts(session_run_id, SPLITS_ARTIFACT, str(tmp_path)))
    assert splits.read_bytes() == encode_split_records(())


def test_a_session_that_raises_is_failed_and_holds_its_failure_record(
    client: MlflowClient, runs: MlflowComparisonRuns, named_run: NamedRun, tmp_path: Path
) -> None:
    context = ComparisonBatchFailureContext(
        phase=ComparisonBatchPhase.WARMUP, case_config_path=Path("case.toml")
    )
    with pytest.raises(_InjectedFailure), runs.session(named_run, {}) as session:
        session_run_id = session.run_id
        error = _InjectedFailure("warm-up failed")
        session.failures.record(context, error)
        raise error

    stored = _all_runs(client, runs)
    assert list(stored) == [session_run_id]
    assert stored[session_run_id].info.status == "FAILED"
    record = Path(
        client.download_artifacts(session_run_id, FAILURE_CONTEXT_ARTIFACT, str(tmp_path))
    )
    assert json.loads(record.read_text(encoding="utf-8")) == {
        "phase": "warmup",
        "exception_type": "_InjectedFailure",
        "message": "warm-up failed",
        "scope": "batch",
        "case_config_path": "case.toml",
    }
