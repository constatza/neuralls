"""Shared helpers for tests that drive the comparison batch against a mocked MLflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import tomli_w

from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind

MLFLOW_MODULE = "neuralls.composition.assignments.comparison_batch.mlflow"
COMPARISON_TRACKING_MLFLOW_MODULE = "neuralls.platform.tracking.comparison_tracking.mlflow"
SETUP_TRACKING = "neuralls.composition.assignments.comparison_batch.setup_comparison_tracking"


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


def write_experiments_config(
    path: Path,
    *,
    with_comparisons: bool = False,
    comparison_name: str = "Comparisons",
) -> None:
    payload: dict[str, object] = {
        "mlflow": {"tracking_uri": build_sqlite_tracking_uri(path.parent / "mlruns" / "mlflow.db")},
        "names": {
            "training": "neuralls-training",
            "comparison": comparison_name,
        },
        "comparison_baseline": {
            "rtol": 1.0e-6,
            "atol": 1.0e-14,
            "max_iterations": 10,
            "stopping_criterion": "residual_norm",
            "m_max": 20,
            "preconditioners": [{"name": "identity", "type": "identity"}],
        },
    }
    if with_comparisons:
        (path.parent / "datasets").mkdir(exist_ok=True)
        write_dataset_config(path.parent / "datasets" / "solutions.toml", "solutions")
        write_dataset_config(path.parent / "datasets" / "gaussian-rhs.toml", "gaussian-rhs")
        payload["datasets"] = [
            {"id": "solutions", "path": "datasets/solutions.toml"},
            {"id": "gaussian-rhs", "path": "datasets/gaussian-rhs.toml"},
        ]
        payload["comparisons"] = [
            {"id": "a", "matrix_dataset": "solutions", "rhs_source": {"kind": "gaussian"}},
            {"id": "b", "matrix_dataset": "solutions", "rhs_source": {"kind": "gaussian"}},
        ]
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)


def mock_cfg(
    *,
    matrix_path: Path | None = None,
    rhs_path: Path | None = None,
    preconditioners: list[PreconditionerConfig] | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.general.data.matrix_path = matrix_path
    cfg.general.data.rhs_path = rhs_path
    cfg.general.data.matrix_index = 0
    cfg.general.data.dataset_alias = None
    cfg.general.data.normalize_system = "matrix"
    cfg.general.data.require_non_residual_rhs = True
    cfg.general.data.selection_seed = None
    cfg.general.data.rhs_source_kind = (
        ComparisonRhsSourceKind.RAW_RHS
        if rhs_path is not None
        else ComparisonRhsSourceKind.GAUSSIAN
    )
    cfg.general.data.rhs_source_params = (
        {"path": rhs_path, "sample_index": 0, "row_kind": RowKind.STANDARD, "scale": 1.0}
        if rhs_path is not None
        else {"mean": 0.0, "std": 1.0}
    )
    cfg.preconditioners = tuple(preconditioners or [])
    cfg.notation_policy = NotationPolicy.default()
    return cfg
