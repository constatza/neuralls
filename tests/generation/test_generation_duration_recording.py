"""Generation duration/memory are stamped on the manifest, and survive a skip."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.digest import canonical_digest


@pytest.fixture
def identity() -> StageIdentity:
    """A generation identity with one component, for reuse-skip testing."""
    return StageIdentity.build("generation", {"config": canonical_digest("duration-test")})


def test_fresh_generation_stamps_duration_and_memory(
    tmp_path: Path, source_spec: SourceSpec, dataset_spec: DatasetSpec
) -> None:
    """A real generation records a non-negative duration and peak memory."""
    dataset_dir = build_dataset(source_spec, dataset_spec, str(tmp_path / "dataset"))

    manifest = read_dataset_manifest(dataset_dir)
    assert manifest.generation_duration_seconds is not None
    assert manifest.generation_duration_seconds >= 0
    assert manifest.generation_peak_memory_bytes is not None
    assert manifest.generation_peak_memory_bytes >= 0


def test_skipped_regeneration_leaves_stamped_duration_untouched(
    tmp_path: Path, source_spec: SourceSpec, dataset_spec: DatasetSpec, identity: StageIdentity
) -> None:
    """Reusing a dataset (identity + content digest match) never re-stamps duration."""
    dataset_dir_str = str(tmp_path / "dataset")
    build_dataset(source_spec, dataset_spec, dataset_dir_str, identity=identity)
    first_manifest = read_dataset_manifest(dataset_dir_str)
    assert first_manifest.generation_duration_seconds is not None

    # Second call under the same identity hits the skip path (is_dataset_reusable
    # returns True) — the manifest must not be touched at all.
    build_dataset(source_spec, dataset_spec, dataset_dir_str, identity=identity)
    second_manifest = read_dataset_manifest(dataset_dir_str)

    assert second_manifest.generation_duration_seconds == first_manifest.generation_duration_seconds
    assert (
        second_manifest.generation_peak_memory_bytes == first_manifest.generation_peak_memory_bytes
    )
