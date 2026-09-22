"""`training_identity`: what makes a trained model reusable, and what must not invalidate it."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuralls.composition.identity import training
from neuralls.composition.identity.training import (
    dataset_unchanged_since,
    training_identity,
)
from neuralls.shared.digest import canonical_digest
from tests.composition.conftest import JobWriter


@pytest.fixture(autouse=True)
def dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    """Isolate dataset resolution: the digest is whatever this dict says."""
    state = {"digest": canonical_digest("dataset-v1")}
    monkeypatch.setattr(training, "assignment_dataset_dir", lambda *_: tmp_path / "ds")
    monkeypatch.setattr(training, "current_dataset_digest", lambda _dir: state["digest"])
    return state


def _key(job: Path) -> str:
    return training_identity(
        job_config_path=job,
        data_config_path=job.parent / "data.toml",
        settings=MagicMock(),
    ).key


def test_identical_inputs_give_the_same_key(write_job: JobWriter) -> None:
    assert _key(write_job()) == _key(write_job())


def test_comments_and_experiment_labels_do_not_invalidate(write_job: JobWriter) -> None:
    """Hashing after DLKit loads the job ignores comments and reporting sections."""
    base = _key(write_job())
    assert _key(write_job(comment="# another comment\n\n")) == base
    assert _key(write_job(experiment_name="renamed-experiment")) == base


def test_a_hyperparameter_change_invalidates(write_job: JobWriter) -> None:
    assert _key(write_job(rank=100)) != _key(write_job(rank=10))


def test_an_edit_to_the_referenced_data_profile_invalidates(write_job: JobWriter) -> None:
    """DLKit inlines the profile, so editing that file changes the key."""
    assert _key(write_job(batch_size=64)) != _key(write_job(batch_size=256))


def test_a_changed_dataset_invalidates(write_job: JobWriter, dataset: dict[str, str]) -> None:
    job = write_job()
    before = _key(job)
    dataset["digest"] = canonical_digest("dataset-v2")
    assert _key(job) != before


def test_key_is_independent_of_where_the_configs_live(write_job: JobWriter, tmp_path: Path) -> None:
    """A copied config tree (another host or checkout) keeps its identity."""
    original = write_job()
    moved_root = tmp_path / "moved"
    shutil.copytree(original.parent.parent, moved_root)
    assert _key(moved_root / "jobs" / "job.toml") == _key(original)


def test_dataset_guard_detects_a_dataset_changed_during_training(
    write_job: JobWriter, dataset: dict[str, str]
) -> None:
    identity = training_identity(
        job_config_path=write_job(),
        data_config_path=Path("unused"),
        settings=MagicMock(),
    )
    assert dataset_unchanged_since(identity.tags(), Path("unused"), MagicMock()) is True
    dataset["digest"] = canonical_digest("regenerated-meanwhile")
    assert dataset_unchanged_since(identity.tags(), Path("unused"), MagicMock()) is False
