"""``splits.toml`` round trip and the separation claims a reader must not accept blindly."""

from __future__ import annotations

import tomllib
from pathlib import Path

import mlflow
import pytest
import tomli_w
from mlflow.tracking import MlflowClient

from neuralls.domain.solver.models.split_record import DatasetSplitRecord
from neuralls.platform.tracking.split_records import (
    SPLITS_ARTIFACT,
    decode_split_records,
    download_session_splits,
    encode_split_records,
)
from neuralls.shared.types import SessionRunId


def test_records_survive_a_round_trip(split_records: tuple[DatasetSplitRecord, ...]) -> None:
    assert decode_split_records(encode_split_records(split_records)) == split_records


def test_a_disjoint_claim_over_overlapping_matrices_is_rejected(
    split_records: tuple[DatasetSplitRecord, ...],
) -> None:
    tables = tomllib.loads(encode_split_records(split_records).decode("utf-8"))
    tables["datasets"]["many"]["test_matrix_ids"] = [1]

    with pytest.raises(ValueError, match="both"):
        decode_split_records(tomli_w.dumps(tables).encode("utf-8"))


def test_a_payload_without_datasets_is_rejected() -> None:
    with pytest.raises(TypeError, match="datasets"):
        decode_split_records(b"")


def test_a_session_returns_the_records_it_logged(
    split_records: tuple[DatasetSplitRecord, ...], mlflow_tracking_uri: str, tmp_path: Path
) -> None:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("split-records")
    with mlflow.start_run() as run:
        mlflow.log_text(encode_split_records(split_records).decode("utf-8"), SPLITS_ARTIFACT)

    downloaded = download_session_splits(
        MlflowClient(mlflow_tracking_uri), SessionRunId(run.info.run_id), tmp_path / "download"
    )

    assert downloaded == split_records


def test_a_session_without_splits_is_rejected(mlflow_tracking_uri: str, tmp_path: Path) -> None:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("split-records")
    with mlflow.start_run() as run:
        mlflow.log_text("x", "other.txt")

    with pytest.raises(ValueError, match="no splits.toml"):
        download_session_splits(
            MlflowClient(mlflow_tracking_uri), SessionRunId(run.info.run_id), tmp_path / "download"
        )
