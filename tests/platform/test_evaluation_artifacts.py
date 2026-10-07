"""Tests for eval-only MLflow artifact resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from neuralls.platform.tracking.artifact_access import ArtifactLease
from neuralls.platform.tracking.artifact_selection import (
    CHECKPOINT_ARTIFACT_DIR,
    CONFIG_ARTIFACT_DIR,
    SPLIT_ARTIFACT_DIR,
)
from neuralls.platform.tracking.evaluation_artifacts import (
    CheckpointArtifactError,
    CorruptCheckpointArtifactError,
    MissingCheckpointArtifactError,
    resolve_training_checkpoint,
    resolve_training_config_artifacts,
    resolve_training_evaluation_artifacts,
    resolve_training_split_artifact,
)

RUN_ID = "run-1"
ASSIGNMENT_ID = "assignment-1"
SOURCE_URI = "file:///mlartifacts/run-1/artifacts"
CHECKPOINT_FILE = "best.ckpt"
SPLIT_FILE = "run_4_split.json"
CONFIG_FILE = "job.toml"
WEIGHTS_PAYLOAD = b"weights"
ARTIFACT_IS_DIR = True
ARTIFACT_IS_FILE = False


@dataclass(frozen=True)
class FakeArtifact:
    """Minimal MLflow FileInfo-shaped object."""

    path: str
    is_dir: bool = ARTIFACT_IS_FILE


class FakeArtifactClient:
    """Minimal MLflowClient double for artifact resolution tests."""

    def __init__(self, listings: dict[str, tuple[FakeArtifact, ...]]):
        self._listings = listings

    def list_artifacts(self, run_id: str, path: str) -> list[FakeArtifact]:
        del run_id
        return list(self._listings.get(path, ()))


class FakeLeaseManager:
    """Lease manager double that returns prebuilt local artifact paths."""

    def __init__(self, paths: dict[str, Path], broken_artifacts: set[str] | None = None) -> None:
        self._paths = paths
        self._broken_artifacts = broken_artifacts or set()
        self.file_calls: list[tuple[str, str]] = []
        self.dir_calls: list[tuple[str, str]] = []

    def resolve_file(self, run_id: str, artifact_path: str) -> ArtifactLease:
        self.file_calls.append((run_id, artifact_path))
        return self._resolve(run_id=run_id, artifact_path=artifact_path)

    def resolve_dir(self, run_id: str, artifact_path: str) -> ArtifactLease:
        self.dir_calls.append((run_id, artifact_path))
        return self._resolve(run_id=run_id, artifact_path=artifact_path)

    def _resolve(self, *, run_id: str, artifact_path: str) -> ArtifactLease:
        if artifact_path in self._broken_artifacts:
            raise RuntimeError("lease failed")
        return ArtifactLease(
            path=self._paths[artifact_path],
            run_id=run_id,
            artifact_path=artifact_path,
            source_uri=SOURCE_URI,
            local_copy=False,
        )


def _write_split(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "train": [0, 1],
                "validation": [2],
                "test": [3],
                "predict": [],
            }
        ),
        encoding="utf-8",
    )


def _checkpoint_listing() -> tuple[FakeArtifact, ...]:
    return (FakeArtifact(f"{CHECKPOINT_ARTIFACT_DIR}/{CHECKPOINT_FILE}"),)


def _split_listing() -> tuple[FakeArtifact, ...]:
    return (FakeArtifact(f"{SPLIT_ARTIFACT_DIR}/{SPLIT_FILE}"),)


def test_resolve_training_split_file_validates_single_json(tmp_path: Path) -> None:
    split_file = tmp_path / SPLIT_FILE
    _write_split(split_file)
    client = FakeArtifactClient({SPLIT_ARTIFACT_DIR: _split_listing()})
    leases = FakeLeaseManager({f"{SPLIT_ARTIFACT_DIR}/{SPLIT_FILE}": split_file})

    result = resolve_training_split_artifact(client=client, run_id=RUN_ID, artifact_leases=leases)

    assert result.file == split_file
    assert result.artifact_path == f"{SPLIT_ARTIFACT_DIR}/{SPLIT_FILE}"
    assert leases.file_calls == [(RUN_ID, f"{SPLIT_ARTIFACT_DIR}/{SPLIT_FILE}")]


def test_resolve_training_split_file_rejects_missing_json() -> None:
    client = FakeArtifactClient({SPLIT_ARTIFACT_DIR: ()})
    leases = FakeLeaseManager({})

    with pytest.raises(FileNotFoundError, match="no split JSON artifact"):
        resolve_training_split_artifact(client=client, run_id=RUN_ID, artifact_leases=leases)

    assert leases.file_calls == []


def test_resolve_training_split_file_rejects_multiple_json_files() -> None:
    client = FakeArtifactClient(
        {
            SPLIT_ARTIFACT_DIR: (
                FakeArtifact(f"{SPLIT_ARTIFACT_DIR}/first.json"),
                FakeArtifact(f"{SPLIT_ARTIFACT_DIR}/second.json"),
            )
        }
    )

    leases = FakeLeaseManager({})

    with pytest.raises(ValueError, match="multiple split JSON artifacts"):
        resolve_training_split_artifact(client=client, run_id=RUN_ID, artifact_leases=leases)

    assert leases.file_calls == []


def test_resolve_training_checkpoint_raises_when_no_checkpoint_artifact() -> None:
    """A FINISHED run with no checkpoint artifact must fail with an actionable error."""
    client = FakeArtifactClient({})
    leases = FakeLeaseManager({})

    with pytest.raises(MissingCheckpointArtifactError, match=f"{RUN_ID}.*{ASSIGNMENT_ID}"):
        resolve_training_checkpoint(
            client=client,
            run_id=RUN_ID,
            artifact_leases=leases,
            assignment_id=ASSIGNMENT_ID,
        )

    assert leases.dir_calls == []


def test_resolve_training_checkpoint_returns_checkpoint_when_present(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / CHECKPOINT_ARTIFACT_DIR
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE
    checkpoint_file.write_bytes(WEIGHTS_PAYLOAD)
    client = FakeArtifactClient({CHECKPOINT_ARTIFACT_DIR: _checkpoint_listing()})
    leases = FakeLeaseManager({CHECKPOINT_ARTIFACT_DIR: checkpoint_dir})

    result = resolve_training_checkpoint(
        client=client,
        run_id=RUN_ID,
        artifact_leases=leases,
        assignment_id=ASSIGNMENT_ID,
    )

    assert result == checkpoint_file
    assert leases.dir_calls == [(RUN_ID, CHECKPOINT_ARTIFACT_DIR)]


def test_resolve_training_checkpoint_raises_corrupt_error_when_lease_fails(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / CHECKPOINT_ARTIFACT_DIR
    checkpoint_dir.mkdir()
    client = FakeArtifactClient({CHECKPOINT_ARTIFACT_DIR: _checkpoint_listing()})
    leases = FakeLeaseManager(
        {CHECKPOINT_ARTIFACT_DIR: checkpoint_dir},
        broken_artifacts={CHECKPOINT_ARTIFACT_DIR},
    )

    with pytest.raises(
        CorruptCheckpointArtifactError, match=f"{RUN_ID}.*{ASSIGNMENT_ID}"
    ) as excinfo:
        resolve_training_checkpoint(
            client=client,
            run_id=RUN_ID,
            artifact_leases=leases,
            assignment_id=ASSIGNMENT_ID,
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_checkpoint_artifact_errors_share_common_base() -> None:
    """Both checkpoint-artifact exceptions must narrow to CheckpointArtifactError."""
    assert issubclass(MissingCheckpointArtifactError, CheckpointArtifactError)
    assert issubclass(CorruptCheckpointArtifactError, CheckpointArtifactError)


def test_resolve_training_config_artifacts_returns_none_when_absent() -> None:
    client = FakeArtifactClient({})
    leases = FakeLeaseManager({})

    result = resolve_training_config_artifacts(client=client, run_id=RUN_ID, artifact_leases=leases)

    assert result.config_dir is None
    assert leases.dir_calls == []


def test_resolve_training_config_artifacts_uses_config_dir_when_present(tmp_path: Path) -> None:
    config_dir = tmp_path / CONFIG_ARTIFACT_DIR
    config_dir.mkdir()
    (config_dir / CONFIG_FILE).write_text("run = {}", encoding="utf-8")
    client = FakeArtifactClient(
        {CONFIG_ARTIFACT_DIR: (FakeArtifact(f"{CONFIG_ARTIFACT_DIR}/{CONFIG_FILE}"),)}
    )

    leases = FakeLeaseManager({CONFIG_ARTIFACT_DIR: config_dir})

    result = resolve_training_config_artifacts(client=client, run_id=RUN_ID, artifact_leases=leases)

    assert result.config_dir == config_dir
    assert leases.dir_calls == [(RUN_ID, CONFIG_ARTIFACT_DIR)]


def test_resolve_training_evaluation_artifacts_resolves_all_required_paths(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / CHECKPOINT_ARTIFACT_DIR
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE
    checkpoint_file.write_bytes(WEIGHTS_PAYLOAD)
    split_file = tmp_path / SPLIT_FILE
    _write_split(split_file)
    config_dir = tmp_path / CONFIG_ARTIFACT_DIR
    config_dir.mkdir()
    client = FakeArtifactClient(
        {
            CHECKPOINT_ARTIFACT_DIR: _checkpoint_listing(),
            SPLIT_ARTIFACT_DIR: _split_listing(),
            CONFIG_ARTIFACT_DIR: (FakeArtifact(f"{CONFIG_ARTIFACT_DIR}/{CONFIG_FILE}"),),
        }
    )

    leases = FakeLeaseManager(
        {
            CHECKPOINT_ARTIFACT_DIR: checkpoint_dir,
            f"{SPLIT_ARTIFACT_DIR}/{SPLIT_FILE}": split_file,
            CONFIG_ARTIFACT_DIR: config_dir,
        }
    )

    result = resolve_training_evaluation_artifacts(
        client=client,
        run_id=RUN_ID,
        artifact_leases=leases,
        assignment_id=ASSIGNMENT_ID,
    )

    assert result.checkpoint_path == checkpoint_file
    assert result.split_file == split_file
    assert result.split_artifact_path == f"{SPLIT_ARTIFACT_DIR}/{SPLIT_FILE}"
    assert result.config_artifacts.config_dir == config_dir
