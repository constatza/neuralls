"""Integration test for MLflow registered-alias model resolution.

Uses local SQLite tracking + local artifacts (no MLflow server process).
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from dlkit.infrastructure.io import url_resolver
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.model_resolution import resolve_model_ref
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.resolution import resolve_local_path
from neuralls.platform.tracking.artifact_access import MlflowArtifactLeaseManager
from neuralls.platform.tracking.model_registry import (
    assign_dataset_alias_to_registered_model,
    register_logged_model,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:codecs\\.open\\(\\) is deprecated.*:DeprecationWarning"),
]


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _artifact_location(path: Path) -> str:
    """Convert a local artifacts root into the URI form expected by MLflow."""
    return url_resolver.build_uri(path.resolve(), scheme="file")


def _artifact_uri_to_path(artifact_uri: str) -> Path:
    """Convert MLflow artifact URI to local filesystem path."""
    return resolve_local_path(artifact_uri, base_dir=Path.cwd())


def _assert_path_within(path: Path, root: Path) -> None:
    """Assert that path is under root, with resolved absolute paths."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise AssertionError(f"Path {resolved_path} is not under {resolved_root}")


def _write_identity_model(path: Path) -> Path:
    """Write a minimal models-from-code pyfunc model script."""
    path.write_text(
        "import mlflow\nfrom typing import Dict, List\n\nclass _IdentityModel(mlflow.pyfunc.PythonModel):\n    def predict(self, context, model_input: List[Dict[str, str]], params=None) -> List[Dict[str, str]]:\n        _ = context\n        _ = params\n        return model_input\n\nmlflow.models.set_model(_IdentityModel())\n",
        encoding="utf-8",
    )
    return path


def test_registered_alias_resolution_with_local_sqlite_tracking(tmp_path: Path) -> None:
    """Assign dataset alias to a registered run and resolve it via model_ref."""
    tracking_uri = _tracking_uri(tmp_path)
    artifacts_dir = (tmp_path / "mlartifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"alias-resolution-integration-{tmp_path.name}"
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=_artifact_location(artifacts_dir),
        )
    mlflow.set_experiment(experiment_name)

    checkpoint_file = tmp_path / "dummy.ckpt"
    checkpoint_file.write_text("dummy-checkpoint")
    model_code = _write_identity_model(tmp_path / "identity_model.py")

    with mlflow.start_run(run_name="tiny-model-run") as run:
        run_id = run.info.run_id
        run_artifact_path = _artifact_uri_to_path(run.info.artifact_uri)
        _assert_path_within(run_artifact_path, artifacts_dir)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_code,
        )
        mlflow.log_artifact(str(checkpoint_file), artifact_path="model")
        mlflow.log_artifact(str(checkpoint_file), artifact_path="checkpoints")

    model_name = "TinyAliasIntegrationModel"
    record = register_logged_model(
        run_id=run_id,
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    registered_version = record.version

    assigned_version = assign_dataset_alias_to_registered_model(
        tracking_uri=tracking_uri,
        registered_model_name=model_name,
        run_id=run_id,
        dataset_alias="solutions",
    )
    assert assigned_version == registered_version

    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name=model_name, alias="@solutions"),
    )

    with MlflowArtifactLeaseManager(client=client) as artifact_leases:
        resolution = resolve_model_ref(
            spec=spec,
            tracking_uri=tracking_uri,
            artifact_leases=artifact_leases,
        )
    assert resolution.run_id == run_id
    assert resolution.checkpoint_path.exists()
    assert resolution.checkpoint_path.suffix == ".ckpt"
    _assert_path_within(resolution.checkpoint_path, artifacts_dir)


def test_experiment_id_registration_resolves_via_latest(tmp_path: Path) -> None:
    """register_logged_model under experiment_id resolves via latest=True without alias."""
    tracking_uri = _tracking_uri(tmp_path)
    artifacts_dir = (tmp_path / "mlartifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"experiment-id-registration-{tmp_path.name}"
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=_artifact_location(artifacts_dir),
        )
    mlflow.set_experiment(experiment_name)

    checkpoint_file = tmp_path / "model.ckpt"
    checkpoint_file.write_text("dummy-checkpoint")
    model_code = _write_identity_model(tmp_path / "identity_model.py")

    with mlflow.start_run(run_name="experiment-id-run") as run:
        run_id = run.info.run_id
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_code,
        )
        mlflow.log_artifact(str(checkpoint_file), artifact_path="model")
        mlflow.log_artifact(str(checkpoint_file), artifact_path="checkpoints")

    experiment_id = "spectral-energy-integration"
    record = register_logged_model(
        run_id=run_id,
        registered_model_name=experiment_id,
        tracking_uri=tracking_uri,
        aliases=("candidate",),
        tags={"model_class": "NormScaledLinearFFNN"},
    )
    assert record.name == experiment_id
    assert record.version == 1

    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(latest=True),
    )

    with MlflowArtifactLeaseManager(client=client) as artifact_leases:
        resolution = resolve_model_ref(
            spec=spec,
            tracking_uri=tracking_uri,
            artifact_leases=artifact_leases,
            model_name=experiment_id,
        )
    assert resolution.run_id == run_id
    assert resolution.checkpoint_path.exists()
    assert resolution.checkpoint_path.suffix == ".ckpt"
    _assert_path_within(resolution.checkpoint_path, artifacts_dir)


def test_two_experiments_same_dataset_no_alias_collision(tmp_path: Path) -> None:
    """Two experiments on the same dataset register as separate models without collision."""
    tracking_uri = _tracking_uri(tmp_path)
    artifacts_dir = (tmp_path / "mlartifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"no-collision-{tmp_path.name}"
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=_artifact_location(artifacts_dir),
        )
    mlflow.set_experiment(experiment_name)

    model_code = _write_identity_model(tmp_path / "identity_model.py")
    checkpoint_file = tmp_path / "model.ckpt"
    checkpoint_file.write_text("dummy-checkpoint")

    run_ids: list[str] = []
    for run_name in ("run-linear", "run-linear-normalized"):
        with mlflow.start_run(run_name=run_name) as run:
            run_ids.append(run.info.run_id)
            mlflow.pyfunc.log_model(artifact_path="model", python_model=model_code)
            mlflow.log_artifact(str(checkpoint_file), artifact_path="model")
            mlflow.log_artifact(str(checkpoint_file), artifact_path="checkpoints")

    record_a = register_logged_model(
        run_id=run_ids[0],
        registered_model_name="spectral-energy-col-test",
        tracking_uri=tracking_uri,
        aliases=("candidate",),
        tags={"model_class": "NormScaledLinearFFNN"},
    )
    record_b = register_logged_model(
        run_id=run_ids[1],
        registered_model_name="spectral-energy-normalized-col-test",
        tracking_uri=tracking_uri,
        aliases=("candidate",),
        tags={"model_class": "NormScaledLinearFFNN"},
    )

    assert record_a.name != record_b.name
    assert record_a.run_id != record_b.run_id
    version_a = client.get_model_version_by_alias(record_a.name, "candidate")
    version_b = client.get_model_version_by_alias(record_b.name, "candidate")
    assert version_a.run_id == run_ids[0]
    assert version_b.run_id == run_ids[1]
