"""The manifest records the dataset's scaling and each matrix's scale factor, and reads them back."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.dataset_support import build_test_dataset

from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import load_dataset_manifest, load_matrix_dense_sample
from neuralls.platform.storage.manifest import DatasetScaling
from neuralls.platform.storage.manifest_io import read_dataset_manifest, save_dataset_manifest
from neuralls.shared.system_scaling import (
    DimensionFactor,
    GershgorinScaling,
    NoScaling,
    SystemScaling,
)
from neuralls.shared.types import MatrixSampleId

_SIZE = 5
_DIAGONAL = 4.0
_GERSHGORIN_BOUND = 6.0
"""Largest absolute row sum of the tridiagonal (1, 4, 1) fixture matrix."""


@pytest.fixture
def temp_matrix_file(tmp_path: Path) -> Path:
    """A 5x5 symmetric positive-definite tridiagonal matrix on disk."""
    matrix_file = tmp_path / "test_matrix.txt"
    matrix = (
        np.diag(np.full(_SIZE, _DIAGONAL))
        + np.diag(np.ones(_SIZE - 1), 1)
        + np.diag(np.ones(_SIZE - 1), -1)
    )
    np.savetxt(matrix_file, matrix)
    return matrix_file


@pytest.fixture
def build_with(temp_matrix_file: Path, tmp_path: Path) -> Callable[[SystemScaling], Path]:
    """Build a one-row dataset with a given scaling and return its directory."""

    def _build(scaling: SystemScaling) -> Path:
        output_dir = tmp_path / f"output_{scaling.kind}"
        build_test_dataset(
            SourceSpec(matrix_path=str(temp_matrix_file)),
            DatasetSpec(
                mixture=MixtureSpec(counts={"neutral_ones": 1}, seed=42, shuffle=False),
                scaling=scaling,
            ),
            str(output_dir),
        )
        return output_dir

    return _build


def test_gershgorin_dataset_records_scaling_and_matrix_alpha(
    build_with: Callable[[SystemScaling], Path],
) -> None:
    output_dir = build_with(GershgorinScaling())
    alpha = _GERSHGORIN_BOUND * np.sqrt(_SIZE)

    scaling = read_dataset_manifest(output_dir).scaling

    assert scaling.scaling == GershgorinScaling()
    assert scaling.matrix_norm_type == "spectral"
    assert dict(scaling.matrix_scales) == {0: pytest.approx(alpha, rel=1e-15)}
    np.testing.assert_allclose(
        np.diag(load_matrix_dense_sample(output_dir, 0)), np.full(_SIZE, _DIAGONAL / alpha)
    )


def test_unscaled_dataset_records_unit_alpha(build_with: Callable[[SystemScaling], Path]) -> None:
    output_dir = build_with(NoScaling())

    raw = load_dataset_manifest(output_dir)["scaling"]

    assert (raw["scaling"], raw["matrix_scales"]) == ({"kind": "none"}, {"0": 1.0})


def test_dataset_scaling_round_trips_through_the_manifest(
    build_with: Callable[[SystemScaling], Path], tmp_path: Path
) -> None:
    manifest = read_dataset_manifest(build_with(NoScaling()))
    scaling = DatasetScaling(
        scaling=GershgorinScaling(dimension_factor=DimensionFactor.ONE),
        matrix_norm=0.25,
        matrix_norm_type="inf",
        matrix_scales={MatrixSampleId(3): 2.5, MatrixSampleId(11): 7.0},
    )
    target = tmp_path / "round_trip"
    target.mkdir()

    save_dataset_manifest(target, replace(manifest, scaling=scaling))

    assert read_dataset_manifest(target).scaling == scaling
    assert load_dataset_manifest(target)["scaling"] == {
        "scaling": {"kind": "gershgorin", "dimension_factor": "one"},
        "matrix_norm": 0.25,
        "matrix_norm_type": "inf",
        "matrix_scales": {"3": 2.5, "11": 7.0},
    }
