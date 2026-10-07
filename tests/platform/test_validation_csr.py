"""CSR matrix-directory detection in comparison input validation, for zarr and hdf5 storage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
import zarr
from scipy import sparse as scipy_sparse
from scipy.sparse import csr_array, csr_matrix

from neuralls.platform.sparse_io.components import CsrComponents, to_components
from neuralls.platform.sparse_io.hdf5_store import Hdf5SparseWriter
from neuralls.platform.sparse_io.protocol import SparseLocation
from neuralls.platform.storage.csr_storage import write_csr_matrix_group
from neuralls.platform.storage.validation import validate_comparison_matrix_input

SEED = 7
CSR_GROUP_KEY = "matrix"
HDF5_FILENAME = "dataset.h5"
NOT_LOADABLE = "is not loadable"


@pytest.fixture
def csr_matrices() -> tuple[csr_array, ...]:
    rng = np.random.default_rng(SEED)
    return tuple(
        csr_array(
            cast(csr_matrix, scipy_sparse.random(3, 4, density=0.5, random_state=rng, format="csr"))
        )
        for _ in range(3)
    )


@pytest.fixture
def csr_components(csr_matrices: tuple[csr_array, ...]) -> list[CsrComponents]:
    return [to_components(matrix) for matrix in csr_matrices]


@pytest.fixture
def zarr_csr_group_dir(tmp_path: Path, csr_matrices: tuple[csr_array, ...]) -> Path:
    group_dir = tmp_path / "zarr_csr"
    write_csr_matrix_group("zarr", group_dir, csr_matrices, member_path=CSR_GROUP_KEY)
    return group_dir


@pytest.fixture
def hdf5_csr_dataset_dir(tmp_path: Path, csr_components: Sequence[CsrComponents]) -> Path:
    dataset_dir = tmp_path / "hdf5_csr"
    location = SparseLocation(path=dataset_dir / HDF5_FILENAME, key=CSR_GROUP_KEY)
    Hdf5SparseWriter().write(location, csr_components)
    return dataset_dir


@pytest.fixture
def zarr_dir_without_csr_members(tmp_path: Path) -> Path:
    root = tmp_path / "zarr_no_csr"
    zarr.open_group(str(root), mode="w").create_array("unrelated", data=np.zeros(2))
    return root


@pytest.fixture
def hdf5_file_without_matrix_group(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "hdf5_no_csr"
    dataset_dir.mkdir()
    with h5py.File(dataset_dir / HDF5_FILENAME, "w") as file:
        file.create_dataset("unrelated", data=np.zeros(2))
    return dataset_dir


def test_zarr_csr_group_directory_is_accepted(zarr_csr_group_dir: Path) -> None:
    validate_comparison_matrix_input(zarr_csr_group_dir)


def test_hdf5_csr_dataset_directory_is_accepted(hdf5_csr_dataset_dir: Path) -> None:
    validate_comparison_matrix_input(hdf5_csr_dataset_dir)


def test_zarr_directory_without_csr_members_is_rejected(
    zarr_dir_without_csr_members: Path,
) -> None:
    with pytest.raises(ValueError, match=NOT_LOADABLE):
        validate_comparison_matrix_input(zarr_dir_without_csr_members)


def test_hdf5_file_without_matrix_group_is_rejected_like_zarr(
    hdf5_file_without_matrix_group: Path,
) -> None:
    with pytest.raises(ValueError, match=NOT_LOADABLE):
        validate_comparison_matrix_input(hdf5_file_without_matrix_group)
