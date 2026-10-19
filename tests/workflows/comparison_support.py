"""Shared helpers for tests that drive the comparison batch against a mocked MLflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from neuralls.platform.config.settings import NeurallsSettings

MLFLOW_MODULE = "neuralls.composition.assignments.comparison_batch.mlflow"
COMPARISON_TRACKING_MLFLOW_MODULE = "neuralls.platform.tracking.comparison_tracking.mlflow"
SETUP_TRACKING = "neuralls.composition.assignments.comparison_batch.setup_comparison_tracking"
LOG_COMPARISON_ARTIFACTS = (
    "neuralls.composition.assignments.comparison_batch.log_comparison_artifacts_to_mlflow"
)


def configure_mock_mlflow(mock_mlflow: MagicMock, run_id: str = "comp-run-id") -> None:
    """Make `start_run` a context manager yielding a run with `run_id`."""
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False
    mock_mlflow.get_artifact_uri.return_value = f"mlartifacts/0/{run_id}/artifacts"


def make_settings(tmp_path: Path) -> NeurallsSettings:
    """Build a NeurallsSettings backed by tmp_path."""
    return NeurallsSettings(
        _env_file=[],
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "output",
    )


def write_dataset_config(path: Path, dataset_id: str) -> None:
    """Write a minimal dataset config stub for registry resolution."""
    path.write_text(f'id = "{dataset_id}"\n\n[output]\n', encoding="utf-8")
