"""Tests for format-overwrite protection and skip-by-default generation in dataset_builder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import (
    _guard_format_conflict,
    _stamp_dataset_identity,
    build_dataset,
    is_dataset_reusable,
)
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.shared.digest import canonical_digest


def _artifact(fmt: str, path: str = "x") -> DatasetArtifact:
    return DatasetArtifact(path=path, format=fmt, dtype="float64", shape=(1,))


@pytest.fixture
def zarr_manifest_dir(tmp_path: Path) -> Path:
    """Dataset directory pre-populated with a zarr manifest."""
    manifest = make_dataset_manifest(
        matrix=_artifact("zarr"),
        rhs=_artifact("zarr"),
        solutions=_artifact("zarr"),
        normalization=DatasetNormalization(
            type="matrix", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
        ),
    )
    save_dataset_manifest(tmp_path, manifest)
    return tmp_path


class TestGuardFormatConflict:
    def test_no_manifest_allows_any_format(self, tmp_path: Path) -> None:
        _guard_format_conflict(tmp_path, "zarr")
        _guard_format_conflict(tmp_path, "npy")

    def test_same_format_allows_overwrite(self, zarr_manifest_dir: Path) -> None:
        _guard_format_conflict(zarr_manifest_dir, "zarr")

    def test_different_format_raises(self, zarr_manifest_dir: Path) -> None:
        with pytest.raises(ValueError, match="zarr.*npy|npy.*zarr"):
            _guard_format_conflict(zarr_manifest_dir, "npy")


@pytest.fixture
def complete_npy_dataset_dir(tmp_path: Path) -> Path:
    """Dataset directory with a manifest whose matrix/rhs/solutions files all exist."""
    for name in ("matrix.npy", "rhs.npy", "solutions.npy"):
        np.save(tmp_path / name, np.zeros(1))
    manifest = make_dataset_manifest(
        matrix=_artifact("npy", "matrix.npy"),
        rhs=_artifact("npy", "rhs.npy"),
        solutions=_artifact("npy", "solutions.npy"),
        normalization=DatasetNormalization(
            type="matrix", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
        ),
    )
    save_dataset_manifest(tmp_path, manifest)
    return tmp_path


@pytest.fixture
def identity() -> StageIdentity:
    """A generation identity with one component."""
    return StageIdentity.build("generation", {"config": canonical_digest("a")})


@pytest.fixture
def stamped_npy_dataset_dir(complete_npy_dataset_dir: Path, identity: StageIdentity) -> Path:
    """Complete dataset whose manifest is stamped with `identity`."""
    _stamp_dataset_identity(complete_npy_dataset_dir, identity)
    return complete_npy_dataset_dir


class TestIsDatasetReusable:
    def test_no_manifest_returns_false(self, tmp_path: Path, identity: StageIdentity) -> None:
        assert is_dataset_reusable(tmp_path, identity) is False

    def test_manifest_with_missing_files_returns_false(
        self, zarr_manifest_dir: Path, identity: StageIdentity
    ) -> None:
        """The manifest references files ('x') that were never written to disk."""
        assert is_dataset_reusable(zarr_manifest_dir, identity) is False

    def test_unstamped_manifest_is_never_trusted_by_existence(
        self, complete_npy_dataset_dir: Path, identity: StageIdentity
    ) -> None:
        assert is_dataset_reusable(complete_npy_dataset_dir, identity) is False

    def test_stamped_manifest_with_same_identity_is_reusable(
        self, stamped_npy_dataset_dir: Path, identity: StageIdentity
    ) -> None:
        assert is_dataset_reusable(stamped_npy_dataset_dir, identity) is True

    def test_different_identity_is_not_reusable(self, stamped_npy_dataset_dir: Path) -> None:
        other = StageIdentity.build("generation", {"config": canonical_digest("b")})
        assert is_dataset_reusable(stamped_npy_dataset_dir, other) is False


class TestBuildDatasetSkipByDefault:
    def test_skips_regeneration_when_dataset_already_complete(
        self, stamped_npy_dataset_dir: Path, identity: StageIdentity
    ) -> None:
        with patch(
            "neuralls.composition.generation.dataset_builder.build_dataset_payload"
        ) as mock_build_payload:
            result = build_dataset(
                SourceSpec(
                    matrix_path="unused.npy",
                ),
                DatasetSpec(),
                str(stamped_npy_dataset_dir),
                dataset_format="npy",
                identity=identity,
            )
        mock_build_payload.assert_not_called()
        assert result == str(stamped_npy_dataset_dir)

    def test_force_regenerates_even_when_dataset_already_complete(
        self, complete_npy_dataset_dir: Path
    ) -> None:
        fake_storage = MagicMock()
        fake_storage.make_accumulator.return_value = MagicMock()
        with (
            patch(
                "neuralls.composition.generation.dataset_builder.build_dataset_payload"
            ) as mock_build_payload,
            patch(
                "neuralls.composition.generation.dataset_builder.make_generation_dataset_storage",
                return_value=fake_storage,
            ),
        ):
            build_dataset(
                SourceSpec(
                    matrix_path="unused.npy",
                ),
                DatasetSpec(),
                str(complete_npy_dataset_dir),
                dataset_format="npy",
                force=True,
            )
        mock_build_payload.assert_called_once()
        fake_storage.write_dataset.assert_called_once()
