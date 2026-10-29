"""The fake and a real local store must resolve the same scenarios identically."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from tests.fakes.mlflow_client import FAKE_EXPERIMENT_NAME, FakeMlflowClient

from neuralls.composition.assignments.model_resolution import resolve_model_ref
from neuralls.platform.config.models.preconditioner import (
    LoggedModelRefConfig,
    ModelRefConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
)
from neuralls.platform.tracking.artifact_access import MlflowArtifactLeaseManager
from neuralls.platform.tracking.model_registry import register_logged_model

type Resolve = Callable[[ModelRefConfig], tuple[str, str]]

REAL_EXPERIMENT_NAME = "contract-experiment"


@dataclass(frozen=True, slots=True)
class Store:
    """One MLflow-like store: seed it, then resolve model refs against it."""

    tracking_uri: str
    experiment_name: str
    client_factory: Callable[..., MlflowClient]
    add_run: Callable[[str, int], str]
    register: Callable[[str, str], int]


def _fake_store(tmp_path: Path) -> Store:
    fake = FakeMlflowClient(artifact_root=tmp_path / "artifacts")
    markers: dict[str, str] = {}

    def add_run(marker: str, start_ms: int) -> str:
        run_id = fake.add_run(marker, start_time_ms=start_ms)
        markers[run_id] = marker
        return run_id

    return Store(
        tracking_uri=fake.tracking_uri,
        experiment_name=FAKE_EXPERIMENT_NAME,
        client_factory=fake,
        add_run=add_run,
        register=lambda name, run_id: fake.register_model_version(
            name, run_id, pinned_path=f"checkpoints/{markers[run_id]}.ckpt"
        ),
    )


def _real_store(tmp_path: Path, mlflow_tracking_uri: str) -> Store:
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment_id = client.create_experiment(REAL_EXPERIMENT_NAME)

    def add_run(marker: str, start_ms: int) -> str:
        checkpoint = tmp_path / f"{marker}.ckpt"
        checkpoint.write_text(marker, encoding="utf-8")
        run = client.create_run(experiment_id, run_name=marker, start_time=start_ms)
        for directory in ("model", "checkpoints"):
            client.log_artifact(run.info.run_id, str(checkpoint), artifact_path=directory)
        client.set_terminated(run.info.run_id)
        return run.info.run_id

    def register(name: str, run_id: str) -> int:
        return register_logged_model(
            run_id=run_id, registered_model_name=name, tracking_uri=mlflow_tracking_uri
        ).version

    return Store(mlflow_tracking_uri, REAL_EXPERIMENT_NAME, MlflowClient, add_run, register)


@pytest.fixture(params=["fake", "real"])
def store(request: pytest.FixtureRequest, tmp_path: Path, mlflow_tracking_uri: str) -> Store:
    """The fake store and a real local SQLite store, seeded through one interface."""
    if request.param == "fake":
        return _fake_store(tmp_path)
    return _real_store(tmp_path, mlflow_tracking_uri)


@pytest.fixture
def resolve(store: Store) -> Iterator[Resolve]:
    """Resolve a model ref to ``(run_id, checkpoint content)`` against the store."""
    with MlflowArtifactLeaseManager(
        client=store.client_factory(tracking_uri=store.tracking_uri)
    ) as leases:

        def _resolve(model_ref: ModelRefConfig) -> tuple[str, str]:
            resolution = resolve_model_ref(
                spec=NeuralPreconditionerConfig(
                    type=PreconditionerType.NEURAL,
                    model_ref=model_ref,
                ),
                tracking_uri=store.tracking_uri,
                artifact_leases=leases,
                client_factory=store.client_factory,
            )
            return resolution.run_id, resolution.checkpoint_path.read_text(encoding="utf-8")

        yield _resolve


def test_latest_registered_version_beats_run_order(store: Store, resolve: Resolve) -> None:
    """The highest version wins even when it backs the earliest-created run."""
    run_a = store.add_run("marker-a", 1_000)
    run_b = store.add_run("marker-b", 2_000)
    store.register("Model", run_b)
    store.register("Model", run_a)

    assert resolve(RegisteredModelRefConfig(name="Model", latest=True)) == (run_a, "marker-a")


def test_latest_logged_run_is_the_most_recently_started(store: Store, resolve: Resolve) -> None:
    """Start time, not creation order, decides which logged run is newest."""
    run_new = store.add_run("marker-new", 5_000)
    store.add_run("marker-old", 1_000)

    ref = LoggedModelRefConfig(latest=True, experiment_name=store.experiment_name)
    assert resolve(ref) == (run_new, "marker-new")


def test_unknown_registered_model_is_a_value_error(
    resolve: Resolve,
) -> None:
    """A missing registered model fails identically in both stores."""
    with pytest.raises(ValueError):
        resolve(RegisteredModelRefConfig(name="Missing", latest=True))
