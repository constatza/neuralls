"""Tests for lease-backed MLflow artifact access."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from dlkit.infrastructure.io import url_resolver

from neuralls.platform.tracking.artifact_access import (
    MlflowArtifactLeaseManager,
    normalize_artifact_path,
)

RUN_ID = "run-1"
REMOTE_ARTIFACT_URI = "http://mlflow.test/artifacts/run-1"
MLFLOW_ARTIFACT_URI = "mlflow-artifacts:/1/run-1/artifacts"
EMPTY_DOWNLOAD_DESTINATION = ""
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = "best.ckpt"
CHECKPOINT_ARTIFACT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_FILE}"
SPLIT_DIR = "splits"
SPLIT_FILE = "split.json"
WEIGHTS_PAYLOAD = b"weights"
EMPTY_JSON = "{}"
REMOTE_ARTIFACT_ROOT_NAME = "remote-artifacts"
LOCAL_ARTIFACT_ROOT_NAME = "artifacts"
MLARTIFACTS_ROOT_NAME = "mlartifacts"
RUN_ARTIFACTS_DIR_NAME = "artifacts"


class FakeMlflowClient:
    """Minimal MLflow client double for artifact lease tests."""

    def __init__(self, artifact_uri: str, artifact_root: Path | None = None) -> None:
        self.artifact_uri = artifact_uri
        self.artifact_root = artifact_root
        self.tracking_uri = "sqlite:////tmp/test-mlflow.db"
        self.download_calls: list[tuple[str, str, str]] = []

    def get_run(self, run_id: str) -> SimpleNamespace:
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id, artifact_uri=self.artifact_uri))

    def download_artifacts(self, run_id: str, path: str, dst_path: str | None = None) -> str:
        self.download_calls.append((run_id, path, dst_path or EMPTY_DOWNLOAD_DESTINATION))
        source = self.artifact_root / path if self.artifact_root is not None else None
        if self.artifact_uri.startswith(("http://", "https://", "mlflow-artifacts:")):
            if source is None:
                raise RuntimeError("artifact_root required for fake downloads")
            if dst_path is None:
                raise RuntimeError("remote fake downloads require dst_path")
            return str(_copy_to_destination(source, path, Path(dst_path)))
        if source is None:
            raise RuntimeError("artifact_root required for fake downloads")
        return str(source)


def _copy_to_destination(source: Path, artifact_path: str, root: Path) -> Path:
    destination = root / artifact_path
    if source.is_dir():
        destination.mkdir(parents=True)
        for child in source.iterdir():
            if child.is_file():
                (destination / child.name).write_bytes(child.read_bytes())
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return destination


def _file_uri(path: Path) -> str:
    return url_resolver.build_uri(path.resolve(), scheme="file")


def _run_artifact_root(tmp_path: Path) -> Path:
    return tmp_path / MLARTIFACTS_ROOT_NAME / RUN_ID / RUN_ARTIFACTS_DIR_NAME


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (CHECKPOINT_ARTIFACT_PATH, CHECKPOINT_ARTIFACT_PATH),
    ],
)
def test_normalize_artifact_path_accepts_relative_paths(raw: str, expected: str) -> None:
    assert normalize_artifact_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "/",
        f"/{CHECKPOINT_ARTIFACT_PATH}",
        rf"\{CHECKPOINT_DIR}\{CHECKPOINT_FILE}",
        rf"{CHECKPOINT_DIR}\{CHECKPOINT_FILE}",
        ".",
        f"{CHECKPOINT_DIR}/../{CHECKPOINT_FILE}",
        f"../{CHECKPOINT_FILE}",
        f"{CHECKPOINT_DIR}/./{CHECKPOINT_FILE}",
    ],
)
def test_normalize_artifact_path_rejects_unsafe_paths(raw: str) -> None:
    with pytest.raises(ValueError, match="artifact_path|Unsafe"):
        normalize_artifact_path(raw)


def test_local_file_uri_artifact_is_borrowed_without_download(tmp_path: Path) -> None:
    artifact_root = _run_artifact_root(tmp_path)
    checkpoint = artifact_root / CHECKPOINT_DIR / CHECKPOINT_FILE
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(WEIGHTS_PAYLOAD)
    client = FakeMlflowClient(_file_uri(artifact_root), artifact_root=artifact_root)

    with MlflowArtifactLeaseManager(client=client) as leases:
        lease = leases.resolve_file(RUN_ID, CHECKPOINT_ARTIFACT_PATH)

    assert lease.path == checkpoint.resolve()
    assert lease.local_copy is False
    assert client.download_calls == []
    assert checkpoint.exists()


def test_local_plain_path_directory_is_borrowed_without_download(tmp_path: Path) -> None:
    artifact_root = _run_artifact_root(tmp_path)
    split_dir = artifact_root / SPLIT_DIR
    split_dir.mkdir(parents=True)
    (split_dir / SPLIT_FILE).write_text(EMPTY_JSON, encoding="utf-8")
    client = FakeMlflowClient(str(artifact_root), artifact_root=artifact_root)

    with MlflowArtifactLeaseManager(client=client) as leases:
        lease = leases.resolve_dir(RUN_ID, SPLIT_DIR)

    assert lease.path == split_dir.resolve()
    assert lease.local_copy is False
    assert client.download_calls == []


def test_remote_file_artifact_is_materialized_and_cleaned(tmp_path: Path) -> None:
    remote_root = tmp_path / REMOTE_ARTIFACT_ROOT_NAME
    checkpoint = remote_root / CHECKPOINT_DIR / CHECKPOINT_FILE
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(WEIGHTS_PAYLOAD)
    client = FakeMlflowClient(REMOTE_ARTIFACT_URI, artifact_root=remote_root)

    with MlflowArtifactLeaseManager(client=client) as leases:
        lease = leases.resolve_file(RUN_ID, CHECKPOINT_ARTIFACT_PATH)
        materialized_path = lease.path
        materialized_root = materialized_path.parents[1]
        assert materialized_path.read_bytes() == WEIGHTS_PAYLOAD
        assert materialized_root.name.startswith("neuralls-mlflow-artifacts-")
        assert lease.local_copy is True

    assert len(client.download_calls) == 1
    assert client.download_calls[0][:2] == (RUN_ID, CHECKPOINT_ARTIFACT_PATH)
    assert client.download_calls[0][2]
    assert not materialized_root.exists()


def test_remote_directory_artifact_is_materialized_and_cleaned(tmp_path: Path) -> None:
    remote_root = tmp_path / REMOTE_ARTIFACT_ROOT_NAME
    split_dir = remote_root / SPLIT_DIR
    split_dir.mkdir(parents=True)
    (split_dir / SPLIT_FILE).write_text(EMPTY_JSON, encoding="utf-8")
    client = FakeMlflowClient(MLFLOW_ARTIFACT_URI, artifact_root=remote_root)

    with MlflowArtifactLeaseManager(client=client) as leases:
        lease = leases.resolve_dir(RUN_ID, SPLIT_DIR)
        materialized_path = lease.path
        assert (materialized_path / SPLIT_FILE).exists()
        assert lease.local_copy is True

    assert not materialized_path.exists()


def test_resolve_requires_active_context(tmp_path: Path) -> None:
    artifact_root = tmp_path / LOCAL_ARTIFACT_ROOT_NAME
    client = FakeMlflowClient(str(artifact_root), artifact_root=artifact_root)
    leases = MlflowArtifactLeaseManager(client=client)

    with pytest.raises(RuntimeError, match="inside a lease-manager context"):
        leases.resolve_file(RUN_ID, CHECKPOINT_ARTIFACT_PATH)


def test_repeated_resolution_reuses_one_lease(tmp_path: Path) -> None:
    remote_root = tmp_path / REMOTE_ARTIFACT_ROOT_NAME
    checkpoint = remote_root / CHECKPOINT_DIR / CHECKPOINT_FILE
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(WEIGHTS_PAYLOAD)
    client = FakeMlflowClient(REMOTE_ARTIFACT_URI, artifact_root=remote_root)

    with MlflowArtifactLeaseManager(client=client) as leases:
        first = leases.resolve_file(RUN_ID, CHECKPOINT_ARTIFACT_PATH)
        second = leases.resolve_file(RUN_ID, CHECKPOINT_ARTIFACT_PATH)

    assert first == second
    assert len(client.download_calls) == 1


def test_missing_local_artifact_fails_without_download(tmp_path: Path) -> None:
    artifact_root = tmp_path / LOCAL_ARTIFACT_ROOT_NAME
    artifact_root.mkdir()
    client = FakeMlflowClient(str(artifact_root), artifact_root=artifact_root)

    with (
        MlflowArtifactLeaseManager(client=client) as leases,
        pytest.raises(FileNotFoundError, match="Expected MLflow artifact file"),
    ):
        leases.resolve_file(RUN_ID, CHECKPOINT_ARTIFACT_PATH)

    assert client.download_calls == []


def test_missing_artifact_uri_fails(tmp_path: Path) -> None:
    class MissingUriClient(FakeMlflowClient):
        def get_run(self, run_id: str) -> SimpleNamespace:
            return SimpleNamespace(info=SimpleNamespace(run_id=run_id, artifact_uri=None))

    client = MissingUriClient(str(tmp_path))

    with (
        MlflowArtifactLeaseManager(client=client) as leases,
        pytest.raises(RuntimeError, match="has no artifact_uri"),
    ):
        leases.resolve_file(RUN_ID, CHECKPOINT_ARTIFACT_PATH)
