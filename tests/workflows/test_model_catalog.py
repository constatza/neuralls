"""Tests for model catalog registration helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.model_registry import (
    CHECKPOINT_ARTIFACT_PATH_TAG,
    assign_dataset_alias_to_registered_model,
    build_registered_model_name,
    register_logged_model,
)
from tests.workflows.conftest import LoggedNamedCheckpointsRunFactory, LoggedRunFactory


def test_build_registered_model_name_uses_architecture_only() -> None:
    """Model registry name should match architecture identifier exactly."""
    assert build_registered_model_name("NormScaledLinearFFNN") == "NormScaledLinearFFNN"


def test_register_logged_model_applies_aliases_and_tags(
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
) -> None:
    """Registration normalizes aliases, pins the checkpoint, and writes version tags."""
    run_id = log_run_with_checkpoint("marker-aliases")

    record = register_logged_model(
        run_id=run_id,
        registered_model_name="NormScaledLinearFFNN",
        tracking_uri=mlflow_tracking_uri,
        aliases=("@solutions", "candidate", "candidate"),
        tags={"dataset": "solutions"},
    )

    assert record.name == "NormScaledLinearFFNN"
    assert record.run_id == run_id
    assert record.model_uri == f"runs:/{run_id}/checkpoints/marker-aliases.ckpt"

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    version_str = str(record.version)
    solutions_version = client.get_model_version_by_alias("NormScaledLinearFFNN", "solutions")
    candidate_version = client.get_model_version_by_alias("NormScaledLinearFFNN", "candidate")
    assert int(solutions_version.version) == record.version
    assert int(candidate_version.version) == record.version

    version = client.get_model_version("NormScaledLinearFFNN", version_str)
    assert version.tags["dataset"] == "solutions"
    assert version.tags[CHECKPOINT_ARTIFACT_PATH_TAG] == "checkpoints/marker-aliases.ckpt"
    datetime.fromisoformat(version.tags["registered_at"])


def test_register_logged_model_rejects_reserved_alias(
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
) -> None:
    """Reserved aliases fail fast before assignment."""
    run_id = log_run_with_checkpoint("marker-reserved")

    with pytest.raises(ValueError, match="reserved"):
        register_logged_model(
            run_id=run_id,
            registered_model_name="NormScaledLinearFFNN",
            tracking_uri=mlflow_tracking_uri,
            aliases=("latest",),
            tags=None,
        )


def test_register_logged_model_raises_when_run_has_no_checkpoints(
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
) -> None:
    """Registration fails loudly when the run has no checkpoints/ artifacts at all."""
    with mlflow.start_run() as run:
        mlflow.log_param("x", 1)
        run_id = run.info.run_id

    with pytest.raises(FileNotFoundError, match="no 'checkpoints' artifacts"):
        register_logged_model(
            run_id=run_id,
            registered_model_name="NoCheckpointModel",
            tracking_uri=mlflow_tracking_uri,
        )


def test_register_logged_model_pins_best_checkpoint_and_tags_source(
    mlflow_tracking_uri: str,
    log_run_with_named_checkpoints: LoggedNamedCheckpointsRunFactory,
) -> None:
    """Registering a run with best+last checkpoints pins the exact best.ckpt artifact."""
    run_id = log_run_with_named_checkpoints(
        {"best.ckpt": "best-content", "last.ckpt": "last-content"}
    )

    record = register_logged_model(
        run_id=run_id,
        registered_model_name="BestCheckpointModel",
        tracking_uri=mlflow_tracking_uri,
    )

    expected_source = f"runs:/{run_id}/checkpoints/best.ckpt"
    assert record.model_uri == expected_source

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    version = client.get_model_version("BestCheckpointModel", str(record.version))
    assert version.source == expected_source
    assert version.tags[CHECKPOINT_ARTIFACT_PATH_TAG] == "checkpoints/best.ckpt"


@patch("neuralls.platform.tracking.model_registry.MlflowClient")
def test_assign_dataset_alias_to_registered_model_picks_highest_version(
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Alias assignment picks highest matching version for the same run."""
    client = mock_client_cls.return_value
    mv1 = MagicMock()
    mv1.version = "2"
    mv1.run_id = "run-1"
    mv2 = MagicMock()
    mv2.version = "7"
    mv2.run_id = "run-1"
    mv_other = MagicMock()
    mv_other.version = "9"
    mv_other.run_id = "run-2"
    client.search_model_versions.return_value = [mv1, mv2, mv_other]
    resolved = MagicMock()
    resolved.version = "7"
    client.get_model_version_by_alias.return_value = resolved

    version = assign_dataset_alias_to_registered_model(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        registered_model_name="NormScaledLinearFFNN",
        run_id="run-1",
        dataset_alias="@solutions",
    )

    assert version == 7
    client.search_model_versions.assert_called_once_with("name='NormScaledLinearFFNN'")
    client.set_registered_model_alias.assert_called_once_with(
        name="NormScaledLinearFFNN",
        alias="solutions",
        version="7",
    )
    client.get_model_version_by_alias.assert_called_once_with(
        name="NormScaledLinearFFNN",
        alias="solutions",
    )


@patch("neuralls.platform.tracking.model_registry.MlflowClient")
def test_assign_dataset_alias_to_registered_model_returns_none_when_missing(
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """No matching version for run_id returns None without alias assignment."""
    client = mock_client_cls.return_value
    mv = MagicMock()
    mv.version = "3"
    mv.run_id = "another-run"
    client.search_model_versions.return_value = [mv]

    version = assign_dataset_alias_to_registered_model(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        registered_model_name="NormScaledLinearFFNN",
        run_id="run-1",
        dataset_alias="solutions",
    )

    assert version is None
    client.set_registered_model_alias.assert_not_called()


def test_register_logged_model_uses_experiment_id_as_name(
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
) -> None:
    """Registering under experiment_id produces a model named by experiment, not architecture."""
    experiment_id = "spectral-energy"
    run_id = log_run_with_checkpoint("marker-exp")

    record = register_logged_model(
        run_id=run_id,
        registered_model_name=experiment_id,
        tracking_uri=mlflow_tracking_uri,
        aliases=("candidate",),
        tags={"model_class": "NormScaledLinearFFNN"},
    )

    assert record.name == experiment_id

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    version = client.get_model_version(experiment_id, str(record.version))
    assert version.tags["model_class"] == "NormScaledLinearFFNN"
    datetime.fromisoformat(version.tags["registered_at"])


def test_register_logged_model_warns_when_name_exists(
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
) -> None:
    """Existing registered names still register a new version and emit one warning."""
    run_id_1 = log_run_with_checkpoint("marker-first")
    run_id_2 = log_run_with_checkpoint("marker-second")

    register_logged_model(
        run_id=run_id_1,
        registered_model_name="exp-1",
        tracking_uri=mlflow_tracking_uri,
        tags={"model_class": "NormScaledLinearFFNN"},
    )

    with patch("neuralls.platform.tracking.model_registry.logger") as mock_logger:
        register_logged_model(
            run_id=run_id_2,
            registered_model_name="exp-1",
            tracking_uri=mlflow_tracking_uri,
            tags={"model_class": "NormScaledLinearFFNN"},
        )
        mock_logger.warning.assert_called_once()


def test_register_logged_model_always_sets_registered_at(
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
) -> None:
    """Every registered model version receives a UTC registration timestamp tag."""
    run_id = log_run_with_checkpoint("marker-timestamp")

    record = register_logged_model(
        run_id=run_id,
        registered_model_name="exp-1",
        tracking_uri=mlflow_tracking_uri,
        tags={"model_class": "NormScaledLinearFFNN"},
    )

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    version = client.get_model_version("exp-1", str(record.version))
    datetime.fromisoformat(version.tags["registered_at"])


def test_register_logged_model_registered_at_not_overridden_by_caller(
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
) -> None:
    """Caller-provided registered_at values must not replace the invariant timestamp."""
    run_id = log_run_with_checkpoint("marker-override")

    record = register_logged_model(
        run_id=run_id,
        registered_model_name="exp-1",
        tracking_uri=mlflow_tracking_uri,
        tags={"registered_at": "override"},
    )

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    version = client.get_model_version("exp-1", str(record.version))
    assert version.tags["registered_at"] != "override"
    datetime.fromisoformat(version.tags["registered_at"])
