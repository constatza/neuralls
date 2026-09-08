"""Tests for scale metadata persistence in data generation."""

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.orchestration import _resolve_final_scale
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.normalization import MatrixScale, load_scale_from_metadata
from neuralls.platform.storage.datasets import (
    load_dataset_manifest,
    load_dense_training_arrays,
    load_matrix_dense_sample,
)


@pytest.fixture
def temp_matrix_file(tmp_path: Path):
    """Create a temporary matrix file."""
    matrix_file = tmp_path / "test_matrix.txt"
    # Create a simple 5x5 symmetric positive definite matrix
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 4.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 4.0],
        ]
    )
    np.savetxt(matrix_file, matrix)
    return matrix_file


def test_scale_metadata_saved_with_matrix_normalization(temp_matrix_file: Path, tmp_path: Path):
    """Test that scale metadata is saved with matrix normalization."""
    output_dir = tmp_path / "output"

    # Build dataset with matrix normalization
    build_dataset(
        SourceSpec(
            matrix_path=str(temp_matrix_file),
            rhs_path=None,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 1},
                seed=42,
                shuffle=False,
                strategy_overrides={"neutral_ones": {"samples": 1}},
            ),
            normalize="matrix",
        ),
        str(output_dir),
    )

    # Load manifest and dense arrays
    manifest = load_dataset_manifest(output_dir)
    _rhs, _ = load_dense_training_arrays(output_dir)

    # Check that metadata fields are present
    assert "normalization" in manifest
    scale_metadata = manifest["normalization"]["scale"]
    assert "spectral_radius_bound" in scale_metadata
    assert "dimension_scale" in scale_metadata

    # Check metadata values
    assert str(manifest["normalization"]["type"]) == "matrix"
    assert isinstance(float(scale_metadata["spectral_radius_bound"]), float)
    assert float(scale_metadata["spectral_radius_bound"]) > 0
    assert isinstance(float(scale_metadata["dimension_scale"]), float)
    assert float(scale_metadata["dimension_scale"]) > 0

    # Test scale reconstruction
    metadata = {
        "spectral_radius_bound": float(scale_metadata["spectral_radius_bound"]),
        "dimension_scale": float(scale_metadata["dimension_scale"]),
    }
    scale = load_scale_from_metadata("matrix", metadata)

    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound == metadata["spectral_radius_bound"]
    assert scale.dimension_scale == metadata["dimension_scale"]
    loaded = load_matrix_dense_sample(output_dir, 0)
    # Pack stores A_norm (zarr writer does not scale back to raw space).
    # Verify the diagonal is in normalized space: A_norm = A_raw / composite_scale.
    composite_scale = float(scale_metadata["spectral_radius_bound"]) * float(
        scale_metadata["dimension_scale"]
    )
    np.testing.assert_allclose(np.diag(loaded), np.full(5, 4.0 / composite_scale), rtol=1e-5)


def test_scale_metadata_not_saved_with_none_normalization(temp_matrix_file: Path, tmp_path: Path):
    """Test that scale metadata is not saved when normalization is 'none'."""
    output_dir = tmp_path / "output_none"

    # Build dataset without normalization
    build_dataset(
        SourceSpec(
            matrix_path=str(temp_matrix_file),
            rhs_path=None,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 1},
                seed=42,
                shuffle=False,
                strategy_overrides={"neutral_ones": {"samples": 1}},
            ),
            normalize="none",
        ),
        str(output_dir),
    )

    manifest = load_dataset_manifest(output_dir)

    # Check that normalize_type is present
    assert "normalization" in manifest
    assert str(manifest["normalization"]["type"]) == "none"

    # Scale parameters should not be present for "none" normalization
    scale_metadata = manifest["normalization"]["scale"]
    assert "spectral_radius_bound" not in scale_metadata
    assert "dimension_scale" not in scale_metadata


def test_multi_matrix_resolution_omits_shared_scale_metadata() -> None:
    """Mixed binding scales drop manifest-level reversible scale metadata."""
    matrix_norm, matrix_value_scale, scale_metadata = _resolve_final_scale(
        norm_values=[1.0, 2.0],
        scale_values=[3.0, 5.0],
        metadata_values=[
            {"spectral_radius_bound": 1.5, "dimension_scale": 2.0},
            {"spectral_radius_bound": 2.5, "dimension_scale": 2.0},
        ],
    )

    assert matrix_norm == 1.0
    assert matrix_value_scale == 1.0
    assert scale_metadata is None
