"""Identity-backed reuse and digests for generated datasets.

Generation skips only when the manifest was stamped with the same generation
identity (dataset config + raw source bytes) AND the artifacts still hash to
the stamped content digest. Copies, renames and format changes of the *stored*
arrays never change the content digest; edits to config or sources do change
the identity.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import neuralls.platform.storage.dataset_digest as dataset_digest_module
from neuralls.composition.generation import dataset_builder
from neuralls.composition.generation.processing import process_config
from neuralls.composition.identity.generation import generation_identity
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.platform.storage.dataset_digest import (
    current_dataset_digest,
    dataset_content_digest,
    stat_digest,
)
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.platform.storage.manifest_io import read_dataset_manifest, save_dataset_manifest
from neuralls.shared.types import DatasetFormat

type ConfigFactory = Callable[..., DataConfigFile]


@pytest.fixture
def payload_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count real generation runs by spying on the payload builder."""
    calls: list[int] = []
    real = dataset_builder.build_dataset_payload

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(dataset_builder, "build_dataset_payload", _spy)
    return calls


def _generate(config: DataConfigFile, *, force: bool = False) -> Path:
    """Mimic the entry point: identity first, then `process_config`."""
    matrix = np.loadtxt(config.source.matrix_path)
    return process_config(config, matrix, force=force, identity=generation_identity(config))


def _bump_mtime(path: Path, *, seconds: float = 5.0) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime + seconds, stat.st_mtime + seconds))


# --- generation identity ------------------------------------------------------


def test_identity_is_stable_for_identical_config_and_sources(data_config: DataConfigFile) -> None:
    assert generation_identity(data_config).key == generation_identity(data_config).key


def test_identity_changes_with_dataset_parameters(make_data_config: ConfigFactory) -> None:
    base = generation_identity(make_data_config(seed=1))
    changed = generation_identity(make_data_config(seed=2))

    assert base.key != changed.key
    assert base.components["config"] != changed.components["config"]
    assert base.components["sources"] == changed.components["sources"]


def test_identity_changes_with_source_content(
    make_data_config: ConfigFactory, spd_matrix_file: Path
) -> None:
    before = generation_identity(make_data_config())
    np.savetxt(spd_matrix_file, np.eye(4) * 7.0)
    after = generation_identity(make_data_config())

    assert before.key != after.key
    assert before.components["config"] == after.components["config"]
    assert before.components["sources"] != after.components["sources"]


def test_identity_ignores_output_dir_and_dataset_id(
    make_data_config: ConfigFactory, tmp_path: Path
) -> None:
    base = generation_identity(make_data_config())
    relabeled = generation_identity(
        make_data_config(dataset_id="renamed", data_dir=tmp_path / "elsewhere")
    )

    assert base.key == relabeled.key


def test_sources_component_does_not_depend_on_source_location(
    make_data_config: ConfigFactory, spd_matrix_file: Path, tmp_path: Path
) -> None:
    """Source bytes are hashed by content; only the config component sees the path string."""
    moved = tmp_path / "moved" / spd_matrix_file.name
    moved.parent.mkdir()
    shutil.copyfile(spd_matrix_file, moved)

    original = generation_identity(make_data_config())
    relocated = generation_identity(make_data_config(matrix_path=moved))

    assert original.components["sources"] == relocated.components["sources"]


def test_missing_declared_source_raises_clear_error(
    make_data_config: ConfigFactory, tmp_path: Path
) -> None:
    config = make_data_config(matrix_path=tmp_path / "absent.txt")

    with pytest.raises(FileNotFoundError, match="absent.txt"):
        generation_identity(config)


# --- reuse semantics ----------------------------------------------------------


def test_identical_config_and_sources_are_reused(
    data_config: DataConfigFile, payload_calls: list[int]
) -> None:
    _generate(data_config)
    _generate(data_config)

    assert len(payload_calls) == 1


def test_changed_dataset_params_regenerate_under_same_id(
    make_data_config: ConfigFactory, payload_calls: list[int]
) -> None:
    _generate(make_data_config(seed=1))
    _generate(make_data_config(seed=2))

    assert len(payload_calls) == 2


def test_changed_source_content_regenerates(
    make_data_config: ConfigFactory, spd_matrix_file: Path, payload_calls: list[int]
) -> None:
    _generate(make_data_config())
    np.savetxt(spd_matrix_file, np.eye(4) * 7.0)
    _generate(make_data_config())

    assert len(payload_calls) == 2


def test_dataset_id_is_only_a_label_of_the_directory(
    make_data_config: ConfigFactory, payload_calls: list[int]
) -> None:
    """The id names the directory, so a new id is a new dataset (generated once),
    but it never leaks into the identity key itself."""
    first = _generate(make_data_config(dataset_id="a"))
    second = _generate(make_data_config(dataset_id="b"))

    assert first != second
    assert len(payload_calls) == 2
    assert read_dataset_manifest(first).identity_key == read_dataset_manifest(second).identity_key


def test_force_regenerates_even_when_reusable(
    data_config: DataConfigFile, payload_calls: list[int]
) -> None:
    _generate(data_config)
    _generate(data_config, force=True)

    assert len(payload_calls) == 2


def test_manifest_without_identity_key_is_regenerated(
    data_config: DataConfigFile, payload_calls: list[int]
) -> None:
    """Legacy manifest / crash between write and stamp: never trusted by existence."""
    dataset_dir = _generate(data_config)
    manifest = read_dataset_manifest(dataset_dir)
    save_dataset_manifest(dataset_dir, replace(manifest, identity_key=None))

    _generate(data_config)

    assert len(payload_calls) == 2
    assert read_dataset_manifest(dataset_dir).identity_key is not None


def test_fresh_write_stamps_identity_and_digests(data_config: DataConfigFile) -> None:
    dataset_dir = _generate(data_config)
    manifest = read_dataset_manifest(dataset_dir)

    assert manifest.identity_key == generation_identity(data_config).key
    assert manifest.identity_components == dict(generation_identity(data_config).components)
    assert manifest.content_digest == dataset_content_digest(dataset_dir)
    assert manifest.stat_digest == stat_digest(dataset_dir)


def test_edited_artifact_is_not_reused(
    data_config: DataConfigFile, payload_calls: list[int]
) -> None:
    dataset_dir = _generate(data_config)
    rhs = resolve_dataset_artifacts(dataset_dir).rhs.path
    np.save(rhs, np.load(rhs) + 1.0)

    _generate(data_config)

    assert len(payload_calls) == 2


# --- digests ------------------------------------------------------------------


def test_copied_dataset_has_same_current_digest(
    generated_dataset_dir: Path, tmp_path: Path
) -> None:
    """A copy with fresh mtimes fails the stat check and is re-hashed to the same digest."""
    copy = tmp_path / "copy"
    shutil.copytree(generated_dataset_dir, copy, copy_function=shutil.copyfile)

    stamped = read_dataset_manifest(generated_dataset_dir).content_digest
    assert current_dataset_digest(copy) == stamped
    assert dataset_content_digest(copy) == dataset_content_digest(generated_dataset_dir)


def test_stat_digest_is_path_independent(generated_dataset_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "elsewhere" / "copy"
    shutil.copytree(generated_dataset_dir, copy)  # copy2 preserves mtimes

    assert stat_digest(copy) == stat_digest(generated_dataset_dir)


def test_content_digest_is_identical_across_formats(
    tmp_path: Path, build_in: Callable[[Path, DatasetFormat], Path]
) -> None:
    digests = {
        fmt: dataset_content_digest(build_in(tmp_path / f"as_{fmt}", fmt))
        for fmt in ("npy", "hdf5", "zarr")
    }

    assert len(set(digests.values())) == 1


def test_fast_path_returns_stamped_digest_without_rehashing(
    generated_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamped = read_dataset_manifest(generated_dataset_dir).content_digest

    def _boom(_: Path) -> str:
        raise AssertionError("content must not be re-hashed when stats are unchanged")

    monkeypatch.setattr(dataset_digest_module, "dataset_content_digest", _boom)

    assert current_dataset_digest(generated_dataset_dir) == stamped


def test_edit_with_changed_stat_changes_current_digest(npy_dataset_dir: Path) -> None:
    stamped = read_dataset_manifest(npy_dataset_dir).content_digest
    rhs = resolve_dataset_artifacts(npy_dataset_dir).rhs.path
    np.save(rhs, np.load(rhs) + 1.0)
    _bump_mtime(rhs)

    assert current_dataset_digest(npy_dataset_dir) != stamped


def test_same_size_edit_with_restored_mtime_is_the_documented_fast_path_blind_spot(
    npy_dataset_dir: Path,
) -> None:
    """Known limitation: identical size and mtime_ns skip re-hashing (`--force` covers it)."""
    stamped = read_dataset_manifest(npy_dataset_dir).content_digest
    rhs = resolve_dataset_artifacts(npy_dataset_dir).rhs.path
    before = rhs.stat()
    np.save(rhs, np.load(rhs) + 1.0)
    os.utime(rhs, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert current_dataset_digest(npy_dataset_dir) == stamped
    assert dataset_content_digest(npy_dataset_dir) != stamped


def test_legacy_manifest_without_digests_is_recomputed(generated_dataset_dir: Path) -> None:
    manifest = read_dataset_manifest(generated_dataset_dir)
    save_dataset_manifest(
        generated_dataset_dir, replace(manifest, content_digest=None, stat_digest=None)
    )

    reloaded = read_dataset_manifest(generated_dataset_dir)
    assert reloaded.content_digest is None
    assert current_dataset_digest(generated_dataset_dir) == manifest.content_digest
