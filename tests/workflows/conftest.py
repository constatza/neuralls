"""Shared pytest fixtures for workflow tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import mlflow
import pytest
from dlkit.infrastructure.config.factories import load_job
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.io import url_resolver
from mlflow.tracking import MlflowClient

from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)

_MINIMAL_TRAINING_JOB_TOML = """
[run]
type = "train"

[model]
name = "ScaleEquivariantEmbeddedFactorizedFFNN"
num_layers = 1
activation = "gelu"

[data]
name = "FlexibleDataset"
batch_size = 4

[data.module]
name = "ArrayDataModule"
module_path = "dlkit.engine.adapters.lightning.datamodules"

[training]
"""

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

EXP_ID_ALPHA: str = "alpha-experiment"
EXP_ID_BETA: str = "beta-experiment"

_IDENTITY_MODEL_SOURCE = "import mlflow\nfrom typing import Dict, List\n\nclass _IdentityModel(mlflow.pyfunc.PythonModel):\n    def predict(self, context, model_input: List[Dict[str, str]], params=None) -> List[Dict[str, str]]:\n        _ = context\n        _ = params\n        return model_input\n\nmlflow.models.set_model(_IdentityModel())\n"


def _artifact_location(path: Path) -> str:
    """Convert a local artifacts root into the file:// URI form MLflow expects.

    Args:
        path: Local directory to use as an experiment artifact root.

    Returns:
        A ``file://`` URI string.
    """
    return url_resolver.build_uri(path.resolve(), scheme="file")


class LoggedRunFactory(Protocol):
    """Callable that logs one MLflow run with a checkpoint marker artifact."""

    def __call__(
        self,
        marker: str,
        *,
        run_name: str | None = None,
        start_time_ms: int | None = None,
    ) -> str: ...


@pytest.fixture
def minimal_training_job(tmp_path: Path) -> TrainingJobConfig:
    """Self-contained training job config, independent of any repo config file."""
    job_path = tmp_path / "minimal-job.toml"
    job_path.write_text(_MINIMAL_TRAINING_JOB_TOML, encoding="utf-8")
    job = load_job(job_path)
    assert isinstance(job, TrainingJobConfig)
    return job


@pytest.fixture
def checkpoint_alpha(tmp_path: Path) -> Path:
    """Fake checkpoint file for the alpha experiment."""
    ckpt = tmp_path / "alpha" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    return ckpt


@pytest.fixture
def checkpoint_beta(tmp_path: Path) -> Path:
    """Fake checkpoint file for the beta experiment."""
    ckpt = tmp_path / "beta" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    return ckpt


@pytest.fixture
def jacobi_spec() -> StandardPreconditionerConfig:
    """Non-neural Jacobi preconditioner spec.

    Returns:
        StandardPreconditionerConfig for jacobi.
    """
    return StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)


@pytest.fixture
def neural_spec_with_checkpoint(checkpoint_alpha: Path) -> NeuralPreconditionerConfig:
    """Neural preconditioner spec with an explicit checkpoint_path.

    Args:
        checkpoint_alpha: Concrete checkpoint path.

    Returns:
        NeuralPreconditionerConfig with checkpoint_path set.
    """
    return NeuralPreconditionerConfig(
        name="neural-explicit",
        type=PreconditionerType.NEURAL,
        checkpoint_path=checkpoint_alpha,
    )


@pytest.fixture
def neural_spec_with_experiment() -> NeuralPreconditionerConfig:
    """Neural preconditioner spec with an assignment reference (no checkpoint).

    Returns:
        NeuralPreconditionerConfig with assignment set but checkpoint_path=None.
    """
    return NeuralPreconditionerConfig(
        name="neural-by-experiment",
        type=PreconditionerType.NEURAL,
        assignment=EXP_ID_ALPHA,
    )


@pytest.fixture
def neural_spec_invalid() -> NeuralPreconditionerConfig:
    """Neural preconditioner spec missing both checkpoint_path and assignment.

    Returns:
        NeuralPreconditionerConfig that should fail validation.
    """
    return NeuralPreconditionerConfig(
        name="neural-invalid",
        type=PreconditionerType.NEURAL,
    )


# ---------------------------------------------------------------------------
# Real local MLflow tracking store fixtures (SQLite + filesystem artifacts)
# ---------------------------------------------------------------------------


@pytest.fixture
def mlflow_tracking_uri(tmp_path: Path) -> str:
    """Local SQLite MLflow tracking URI scoped to this test's temp directory.

    Args:
        tmp_path: Pytest-provided isolated temp directory.

    Returns:
        A ``sqlite:///...`` tracking URI string.
    """
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture
def mlflow_experiment(tmp_path: Path, mlflow_tracking_uri: str) -> str:
    """Create and activate a real local MLflow experiment with file-backed artifacts.

    Args:
        tmp_path: Pytest-provided isolated temp directory.
        mlflow_tracking_uri: Local SQLite tracking URI to bind the experiment to.

    Returns:
        The created experiment's name.
    """
    artifacts_dir = (tmp_path / "mlartifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name = f"workflow-test-{tmp_path.name}"
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=_artifact_location(artifacts_dir),
        )
    mlflow.set_experiment(experiment_name)
    return experiment_name


@pytest.fixture
def log_run_with_checkpoint(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
) -> LoggedRunFactory:
    """Factory fixture: log one real run with a minimal pyfunc model + checkpoint.

    Each call logs a distinct run under ``mlflow_experiment`` containing a
    ``.ckpt`` artifact under both the ``model`` and ``checkpoints`` artifact
    paths whose content equals ``marker``, so callers can assert exactly which
    run's checkpoint was resolved. Logging under ``checkpoints`` lets the same
    fixture back both raw-run-reference resolution (``model`` fallback) and
    registration (which requires a ``checkpoints/`` artifact to pin).
    ``start_time_ms`` optionally overrides the run's start time to make
    ordering-sensitive assertions (e.g. "latest run") deterministic instead of
    relying on real wall-clock ordering.

    Args:
        tmp_path: Pytest-provided isolated temp directory.
        mlflow_tracking_uri: Local SQLite tracking URI.
        mlflow_experiment: Active experiment name (ensures setup ordering).

    Returns:
        A callable ``(marker, *, run_name=None, start_time_ms=None) -> run_id``.
    """
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_experiment)
    if experiment is None:
        raise RuntimeError(f"Experiment '{mlflow_experiment}' was not found after creation.")

    def _log_run(
        marker: str,
        *,
        run_name: str | None = None,
        start_time_ms: int | None = None,
    ) -> str:
        checkpoint_file = tmp_path / f"{marker}.ckpt"
        checkpoint_file.write_text(marker)
        model_code = tmp_path / f"{marker}_model.py"
        model_code.write_text(_IDENTITY_MODEL_SOURCE)

        if start_time_ms is not None:
            created = client.create_run(
                experiment_id=experiment.experiment_id,
                run_name=run_name or marker,
                start_time=start_time_ms,
            )
            run_context = mlflow.start_run(run_id=created.info.run_id)
        else:
            run_context = mlflow.start_run(run_name=run_name or marker)

        with run_context as run:
            mlflow.pyfunc.log_model(artifact_path="model", python_model=model_code)
            mlflow.log_artifact(str(checkpoint_file), artifact_path="model")
            mlflow.log_artifact(str(checkpoint_file), artifact_path="checkpoints")
            return run.info.run_id

    return _log_run


class LoggedNamedCheckpointsRunFactory(Protocol):
    """Callable that logs one MLflow run with several named checkpoint files."""

    def __call__(self, checkpoints: Mapping[str, str]) -> str: ...


@pytest.fixture
def log_run_with_named_checkpoints(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
) -> LoggedNamedCheckpointsRunFactory:
    """Factory fixture: log one real run with several named checkpoint files.

    Each call logs a distinct run under ``mlflow_experiment`` with one
    ``checkpoints/<name>`` artifact per ``(name, content)`` pair given —
    e.g. ``{"best.ckpt": "best", "last.ckpt": "last"}`` — so callers can
    exercise ambiguous-checkpoint selection and registration pinning against
    multiple real candidate files.

    Args:
        tmp_path: Pytest-provided isolated temp directory.
        mlflow_tracking_uri: Local SQLite tracking URI.
        mlflow_experiment: Active experiment name (ensures setup ordering).

    Returns:
        A callable ``(checkpoints: Mapping[str, str]) -> run_id``.
    """

    def _log_run(checkpoints: Mapping[str, str]) -> str:
        with mlflow.start_run() as run:
            run_dir = tmp_path / run.info.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            for name, content in checkpoints.items():
                checkpoint_file = run_dir / name
                checkpoint_file.write_text(content)
                mlflow.log_artifact(str(checkpoint_file), artifact_path="checkpoints")
            return run.info.run_id

    return _log_run
