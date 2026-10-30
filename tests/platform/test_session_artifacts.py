from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.reporting.benchmark_events import (
    METHOD_ARTIFACT_DIR,
    MethodBenchmarkArtifacts,
)
from neuralls.platform.reporting.stored_benchmark import read_stored_benchmark
from neuralls.platform.tracking.session_artifacts import (
    PARENT_RUN_TAG,
    MethodArtifactsMissingError,
    download_session_methods,
    session_method_runs,
)

EXPERIMENT = "Comparisons-Case"

type LogMethod = Callable[..., str]


@pytest.fixture
def client(mlflow_tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=mlflow_tracking_uri)


@pytest.fixture
def new_session(client: MlflowClient) -> Callable[[], str]:
    """Create a FINISHED session run in the comparison experiment."""
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)

    def create() -> str:
        run = client.create_run(experiment_id, tags={"phase": "session_comparison"})
        client.set_terminated(run.info.run_id, status="FINISHED")
        return run.info.run_id

    return create


@pytest.fixture
def log_method(
    client: MlflowClient, tmp_path: Path, write_method: Callable[..., CompletedMethodWorkload]
) -> LogMethod:
    """Log a method child under a session holding the named artifacts of a written workload."""

    def log(
        session_run_id: str,
        kind: PreconditionerType,
        *,
        names: Iterable[str] = MethodBenchmarkArtifacts.file_names(),
        status: str = "FINISHED",
    ) -> str:
        source = tmp_path / f"source-{kind}"
        write_method(kind, source)
        experiment_id = client.get_run(session_run_id).info.experiment_id
        run = client.create_run(
            experiment_id, tags={"phase": "comparison_method", PARENT_RUN_TAG: session_run_id}
        )
        for name in names:
            client.log_artifact(
                run.info.run_id, str(source / name), artifact_path=METHOD_ARTIFACT_DIR
            )
        client.set_terminated(run.info.run_id, status=status)
        return run.info.run_id

    return log


def test_session_methods_download_into_a_readable_benchmark(
    client: MlflowClient,
    tmp_path: Path,
    new_session: Callable[[], str],
    log_method: LogMethod,
) -> None:
    session = new_session()
    jacobi_run = log_method(session, PreconditionerType.JACOBI)
    identity_run = log_method(session, PreconditionerType.IDENTITY)
    log_method(new_session(), PreconditionerType.ILU)

    directories = download_session_methods(client, session, tmp_path / "download")

    stored = read_stored_benchmark(directories)
    assert {str(method.name): method.run_id for method in stored.methods} == {
        "jacobi": jacobi_run,
        "identity": identity_run,
    }


def test_only_finished_method_children_of_the_session_are_found(
    client: MlflowClient, new_session: Callable[[], str], log_method: LogMethod
) -> None:
    session = new_session()
    finished = log_method(session, PreconditionerType.JACOBI)
    log_method(session, PreconditionerType.IDENTITY, status="FAILED")

    assert [run.info.run_id for run in session_method_runs(client, session)] == [finished]


def test_a_method_missing_an_artifact_is_named(
    client: MlflowClient, tmp_path: Path, new_session: Callable[[], str], log_method: LogMethod
) -> None:
    session = new_session()
    names = [n for n in MethodBenchmarkArtifacts.file_names() if n != "presentation.toml"]
    broken = log_method(session, PreconditionerType.JACOBI, names=names)

    with pytest.raises(MethodArtifactsMissingError) as raised:
        download_session_methods(client, session, tmp_path / "download")

    assert (raised.value.run_id, raised.value.missing) == (broken, ["presentation.toml"])


def test_a_session_without_methods_is_rejected(
    client: MlflowClient, tmp_path: Path, new_session: Callable[[], str]
) -> None:
    with pytest.raises(ValueError, match="no finished method runs"):
        download_session_methods(client, new_session(), tmp_path / "download")
