"""Tests for MLflow artifact selection policy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from neuralls.platform.tracking.artifact_selection import (
    CONFIG_ARTIFACT_DIR,
    SPLIT_ARTIFACT_DIR,
    has_artifact_dir,
    list_artifact_files,
    select_config_artifact_dir,
    select_split_artifact_path,
)

RUN_ID = "run-1"
SPLIT_FILE_A = f"{SPLIT_ARTIFACT_DIR}/a.json"
SPLIT_FILE_B = f"{SPLIT_ARTIFACT_DIR}/b.json"
NON_JSON_SPLIT_FILE = f"{SPLIT_ARTIFACT_DIR}/notes.txt"
CONFIG_FILE = f"{CONFIG_ARTIFACT_DIR}/job.toml"
EMPTY_ARTIFACTS: tuple[()] = ()


@dataclass(frozen=True)
class FakeArtifact:
    """Minimal MLflow FileInfo-shaped object."""

    path: str
    is_dir: bool = False


class FakeMlflowClient:
    """Minimal MLflow client double for artifact selection tests."""

    def __init__(self, listings: dict[str, tuple[FakeArtifact, ...]]) -> None:
        self._listings = listings
        self.list_calls: list[tuple[str, str]] = []

    def list_artifacts(self, run_id: str, path: str) -> list[FakeArtifact]:
        self.list_calls.append((run_id, path))
        return list(self._listings.get(path, EMPTY_ARTIFACTS))


def test_list_artifact_files_filters_directories() -> None:
    client = FakeMlflowClient(
        {
            SPLIT_ARTIFACT_DIR: (
                FakeArtifact(SPLIT_FILE_A),
                FakeArtifact(f"{SPLIT_ARTIFACT_DIR}/nested", is_dir=True),
            )
        }
    )

    result = list_artifact_files(client, run_id=RUN_ID, artifact_dir=SPLIT_ARTIFACT_DIR)

    assert result == (SPLIT_FILE_A,)
    assert client.list_calls == [(RUN_ID, SPLIT_ARTIFACT_DIR)]


def test_select_split_artifact_path_returns_single_json() -> None:
    client = FakeMlflowClient(
        {SPLIT_ARTIFACT_DIR: (FakeArtifact(NON_JSON_SPLIT_FILE), FakeArtifact(SPLIT_FILE_A))}
    )

    result = select_split_artifact_path(client, run_id=RUN_ID)

    assert result == SPLIT_FILE_A


def test_select_split_artifact_path_rejects_missing_json() -> None:
    client = FakeMlflowClient({SPLIT_ARTIFACT_DIR: (FakeArtifact(NON_JSON_SPLIT_FILE),)})

    with pytest.raises(FileNotFoundError, match="no split JSON artifact"):
        select_split_artifact_path(client, run_id=RUN_ID)


def test_select_split_artifact_path_rejects_multiple_json_files() -> None:
    client = FakeMlflowClient(
        {SPLIT_ARTIFACT_DIR: (FakeArtifact(SPLIT_FILE_B), FakeArtifact(SPLIT_FILE_A))}
    )

    with pytest.raises(ValueError, match="multiple split JSON artifacts"):
        select_split_artifact_path(client, run_id=RUN_ID)


def test_select_config_artifact_dir_returns_dir_when_present() -> None:
    client = FakeMlflowClient({CONFIG_ARTIFACT_DIR: (FakeArtifact(CONFIG_FILE),)})

    result = select_config_artifact_dir(client, run_id=RUN_ID)

    assert result == CONFIG_ARTIFACT_DIR


def test_select_config_artifact_dir_returns_none_when_absent() -> None:
    client = FakeMlflowClient({})

    result = select_config_artifact_dir(client, run_id=RUN_ID)

    assert result is None


def test_has_artifact_dir_rejects_unsafe_dir_before_listing() -> None:
    client = FakeMlflowClient({})

    with pytest.raises(ValueError, match="Unsafe"):
        has_artifact_dir(client, run_id=RUN_ID, artifact_dir="../config")

    assert client.list_calls == []
