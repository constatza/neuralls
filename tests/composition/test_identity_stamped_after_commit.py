"""The identity stamp and content digest are written only after the streamed dataset is committed.

The event order is recorded through two seams: the commit rename in ``finalize`` and the
stamp function in ``dataset_builder``. Fixtures live in ``tests/composition/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from neuralls.composition.generation import dataset_builder, finalize
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.types import DatasetFormat

_INTERRUPT_AFTER_BATCHES = 1
_RENAME = "rename"
_STAMP = "stamp"


@pytest.fixture
def stage_identity() -> StageIdentity:
    return StageIdentity(stage="generation", key="atomic-commit-test", components={"spec": "fixed"})


@pytest.fixture
def event_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record rename and stamp events in the order they happen."""
    events: list[str] = []
    original_rename = finalize._rename_staged
    original_stamp: Callable[..., None] = dataset_builder._stamp_dataset_identity

    def _recording_rename(staging: Path, final: Path) -> None:
        events.append(_RENAME)
        original_rename(staging, final)

    def _recording_stamp(dataset_dir: Path, *args: object, **kwargs: object) -> None:
        events.append(_STAMP)
        original_stamp(dataset_dir, *args, **kwargs)

    monkeypatch.setattr(finalize, "_rename_staged", _recording_rename)
    monkeypatch.setattr(dataset_builder, "_stamp_dataset_identity", _recording_stamp)
    return events


def test_identity_stamped_after_rename(
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
    dense_format: DatasetFormat,
    stage_identity: StageIdentity,
    event_log: list[str],
    tmp_path: Path,
) -> None:
    final = tmp_path / "dataset"
    build_dataset(
        atomic_source,
        atomic_spec,
        str(final),
        dataset_format=dense_format,
        identity=stage_identity,
    )
    assert event_log == [_RENAME, _STAMP]
    manifest = (final / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert stage_identity.key in manifest
    assert '"content_digest"' in manifest


def test_interrupted_run_has_no_identity(
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
    dense_format: DatasetFormat,
    stage_identity: StageIdentity,
    interrupt_writer: Callable[[int], None],
    event_log: list[str],
    tmp_path: Path,
) -> None:
    interrupt_writer(_INTERRUPT_AFTER_BATCHES)
    final = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="injected interruption"):
        build_dataset(
            atomic_source,
            atomic_spec,
            str(final),
            dataset_format=dense_format,
            identity=stage_identity,
        )
    assert event_log == []
    assert not final.exists()
    for manifest in tmp_path.rglob(DATASET_MANIFEST_FILENAME):
        assert stage_identity.key not in manifest.read_text(encoding="utf-8")
