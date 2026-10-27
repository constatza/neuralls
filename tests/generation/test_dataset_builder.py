"""Tests for format-overwrite protection and skip-by-default generation in dataset_builder."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import (
    DatasetProvenanceMismatchError,
    _guard_format_conflict,
    _stamp_dataset_identity,
    is_dataset_reusable,
    write_dense_streamed,
)
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.shared.digest import canonical_digest
from tests.dataset_support import build_test_dataset


def _artifact(fmt: str, path: str = "x", *, rows: int | None = None) -> DatasetArtifact:
    return DatasetArtifact(
        path=path, format=fmt, dtype="float64", shape=(1,), logical_sample_count=rows
    )


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
        with pytest.raises(ValueError):
            _guard_format_conflict(zarr_manifest_dir, "npy")


@pytest.fixture
def complete_npy_dataset_dir(tmp_path: Path, dataset_presentation: DatasetPresentation) -> Path:
    """Dataset directory with a manifest whose matrix/rhs/solutions files all exist."""
    rows = dataset_presentation.generation.generated_rows
    for name in ("matrix.npy", "rhs.npy", "solutions.npy"):
        np.save(tmp_path / name, np.zeros(1))
    manifest = make_dataset_manifest(
        matrix=_artifact("npy", "matrix.npy", rows=rows),
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
def stamped_npy_dataset_dir(
    complete_npy_dataset_dir: Path,
    identity: StageIdentity,
    dataset_presentation: DatasetPresentation,
) -> Path:
    """Complete dataset whose manifest is stamped with `identity`."""
    _stamp_dataset_identity(
        complete_npy_dataset_dir,
        identity,
        generation_usage=None,
        presentation=dataset_presentation,
    )
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


@pytest.fixture
def stamped_hdf5_dataset_dir(
    tmp_path: Path,
    source_spec: SourceSpec,
    dataset_spec: DatasetSpec,
    identity: StageIdentity,
) -> Path:
    """A real hdf5 dataset built under ``identity``, so its manifest is stamped and complete."""
    target = tmp_path / "stamped_hdf5"
    build_test_dataset(
        source_spec, dataset_spec, str(target), dataset_format="hdf5", identity=identity
    )
    return target


class TestBuildDatasetSkipByDefault:
    def test_skips_regeneration_when_dataset_already_complete(
        self, stamped_hdf5_dataset_dir: Path, identity: StageIdentity, source_spec: SourceSpec
    ) -> None:
        with patch(
            "neuralls.composition.generation.dataset_builder.write_dense_streamed"
        ) as writer:
            result = build_test_dataset(
                source_spec,
                DatasetSpec(),
                str(stamped_hdf5_dataset_dir),
                dataset_format="hdf5",
                identity=identity,
            )
        writer.assert_not_called()
        assert result == str(stamped_hdf5_dataset_dir)

    def test_force_regenerates_even_when_dataset_already_complete(
        self,
        stamped_hdf5_dataset_dir: Path,
        source_spec: SourceSpec,
        dataset_spec: DatasetSpec,
    ) -> None:
        with patch(
            "neuralls.composition.generation.dataset_builder.write_dense_streamed",
            wraps=write_dense_streamed,
        ) as writer:
            build_test_dataset(
                source_spec,
                dataset_spec,
                str(stamped_hdf5_dataset_dir),
                dataset_format="hdf5",
                force=True,
            )
        writer.assert_called_once()


class TestStampRefusesInconsistentProvenance:
    def test_rows_recorded_differently_from_the_persisted_count_raise(
        self,
        complete_npy_dataset_dir: Path,
        dataset_presentation: DatasetPresentation,
    ) -> None:
        """Kills: stamping provenance whose rows are not the persisted sample count."""
        generation = replace(
            dataset_presentation.generation,
            generated_rows=dataset_presentation.generation.generated_rows + 1,
            strategies=(
                replace(
                    dataset_presentation.generation.strategies[0],
                    rows=dataset_presentation.generation.generated_rows + 1,
                ),
            ),
        )
        with pytest.raises(DatasetProvenanceMismatchError, match="persisted 100"):
            _stamp_dataset_identity(
                complete_npy_dataset_dir,
                None,
                generation_usage=None,
                presentation=replace(dataset_presentation, generation=generation),
            )

    def test_a_manifest_without_a_sample_count_raises(
        self, tmp_path: Path, dataset_presentation: DatasetPresentation
    ) -> None:
        """Kills: silently dropping provenance when the manifest has no sample count."""
        np.save(tmp_path / "x.npy", np.zeros(1))
        manifest = make_dataset_manifest(
            matrix=_artifact("npy", "x.npy"),
            rhs=_artifact("npy", "x.npy"),
            solutions=_artifact("npy", "x.npy"),
            normalization=DatasetNormalization(
                type="matrix", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
            ),
        )
        save_dataset_manifest(tmp_path, manifest)
        with pytest.raises(DatasetProvenanceMismatchError, match="unknown number"):
            _stamp_dataset_identity(
                tmp_path, None, generation_usage=None, presentation=dataset_presentation
            )
