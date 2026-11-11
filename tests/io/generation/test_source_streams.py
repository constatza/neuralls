from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from tests.dataset_support import build_test_dataset
from tests.generation_support import spd_matrix

from neuralls.domain.generation.source_streams import (
    EnumerateBy,
)
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.dataset_readers import (
    load_matrix_sample_index,
    load_parameter_arrays,
    load_row_kind_codes,
)
from neuralls.platform.storage.datasets import (
    load_dataset_manifest,
    load_dense_training_arrays,
    load_matrix_dense_sample,
    resolve_dataset_paths,
)
from neuralls.platform.storage.enum_codecs import decode_row_kind_array
from neuralls.platform.storage.generation_formats import HDF5_FILENAME as DENSE_DATASET_FILE
from neuralls.shared.types import RowKind


@pytest.fixture
def three_spd_txt_matrices(tmp_path: Path) -> tuple[Path, int]:
    """Three 5×5 SPD matrices written to individual .txt files for glob tests."""
    rng = np.random.default_rng(7)
    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir()
    n = 5
    for i in range(3):
        np.savetxt(mat_dir / f"A_{i:03d}.txt", spd_matrix(rng, n))
    return mat_dir, n


def test_build_dataset_streams_matrix_stack_without_dense_batch(tmp_path: Path) -> None:
    matrix_stack = np.array(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[6.0, 0.0], [0.0, 4.0]],
        ],
        dtype=np.float64,
    )
    matrix_path = tmp_path / "matrix_stack.npy"
    np.save(matrix_path, matrix_stack)

    out_dir = tmp_path / "dataset"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 2},
                seed=42,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_path), mode="r")
    assert mat_arr.shape[0] == 2
    dense0 = load_matrix_dense_sample(out_dir, 0)
    dense1 = load_matrix_dense_sample(out_dir, 1)
    np.testing.assert_allclose(dense0, matrix_stack[0])
    np.testing.assert_allclose(dense1, matrix_stack[1])


def test_build_dataset_hdf5_output_holds_the_generated_matrix(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    out_dir = tmp_path / "dataset_npy"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 2},
                seed=42,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="hdf5",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)

    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)
    np.testing.assert_allclose(load_matrix_dense_sample(out_dir, 0), matrix)


def test_single_matrix_stored_once_in_zarr(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, rhs)

    out_dir = tmp_path / "dataset_single_matrix"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
            rhs_path=str(rhs_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 1},
                seed=7,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    saved_rhs, saved_solutions = load_dense_training_arrays(out_dir)
    assert saved_rhs.shape == (3, 2)
    assert saved_solutions.shape == (3, 2)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_path), mode="r")
    assert mat_arr.shape[0] == 1
    dense_last = load_matrix_dense_sample(out_dir, 0)
    np.testing.assert_allclose(dense_last, matrix)


def test_build_dataset_persists_residuals_pairs(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    for idx in range(3):
        np.savetxt(tmp_path / f"sol_{idx:03d}.txt", np.array([1.0 + idx, 0.5 + idx]))

    out_dir = tmp_path / "residuals_dataset"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"residuals": 4},
                seed=7,
                shuffle=False,
                strategy_overrides={
                    "residuals": {
                        "stop": 2,
                        "start": 1,
                        "solutions_glob": str(tmp_path / "sol_*.txt"),
                    }
                },
            ),
            normalize="none",
        ),
        str(out_dir),
    )

    saved_rhs, saved_solutions = load_dense_training_arrays(out_dir)
    assert saved_rhs.shape == (4, 2)
    assert saved_solutions.shape == (4, 2)
    np.testing.assert_allclose(saved_solutions @ matrix.T, saved_rhs, atol=1e-10)


def test_build_dataset_persists_gaussian_residual_pairs(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    out_dir = tmp_path / "gaussian_residual_dataset"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_residuals": 4},
                seed=7,
                shuffle=False,
                strategy_overrides={"gaussian_residuals": {"stop": 2, "start": 1}},
            ),
            normalize="none",
        ),
        str(out_dir),
    )

    saved_rhs, saved_solutions = load_dense_training_arrays(out_dir)
    assert saved_rhs.shape == (4, 2)
    assert saved_solutions.shape == (4, 2)
    np.testing.assert_allclose(saved_solutions @ matrix.T, saved_rhs, atol=1e-10)


def test_build_dataset_persists_row_metadata_artifacts(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    out_dir = tmp_path / "dataset_meta"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 3},
                seed=42,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="hdf5",
    )

    manifest = load_dataset_manifest(out_dir)
    assert manifest["row_kind"]["path"] == DENSE_DATASET_FILE
    assert manifest["matrix_sample_index"]["path"] == DENSE_DATASET_FILE

    row_kinds = decode_row_kind_array(load_row_kind_codes(out_dir))
    matrix_indices = load_matrix_sample_index(out_dir)

    assert row_kinds == (RowKind.STANDARD, RowKind.STANDARD, RowKind.STANDARD)
    np.testing.assert_array_equal(matrix_indices, np.array([0, 0, 0], dtype=np.int64))


def test_build_dataset_marks_residual_error_rows_with_kind_codes(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    out_dir = tmp_path / "dataset_residual_meta"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_residuals": 2},
                seed=7,
                shuffle=False,
                strategy_overrides={"gaussian_residuals": {"stop": 2, "start": 1}},
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="hdf5",
    )

    row_kinds = decode_row_kind_array(load_row_kind_codes(out_dir))

    # stop=2, start=1 -> K=2 rows (iterates 1, 2), both CG_INTERNAL: iterate 0
    # (the base pair) is never emitted.
    assert row_kinds == (RowKind.CG_INTERNAL, RowKind.CG_INTERNAL)


def test_glob_matrix_gaussian_uses_global_sample_budget(
    three_spd_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """Multi-matrix gaussian generation treats counts as a global budget."""
    mat_dir, n = three_spd_txt_matrices
    total_samples = 4  # not a multiple of three: one matrix gets 2, the others 1 each
    out_dir = tmp_path / "ds_gaussian"

    build_test_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "A_*.txt"),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": total_samples},
                seed=0,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (total_samples, n)
    assert solutions.shape == (total_samples, n)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_path), mode="r")
    assert mat_arr.shape[0] == total_samples

    per_matrix = np.bincount(load_matrix_sample_index(out_dir), minlength=3)
    assert sorted(per_matrix.tolist()) == [1, 1, 2]


def test_glob_matrix_solution_archive_uses_global_sample_budget(
    three_spd_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """Multi-matrix solution_archive treats counts as a global budget."""
    mat_dir, n = three_spd_txt_matrices
    sol_dir = tmp_path / "solutions"
    sol_dir.mkdir()
    rng = np.random.default_rng(13)
    n_solutions = 5  # 3 matrices x 5 files: the cyclic map gives 2, 2 and 1 units
    for j in range(n_solutions):
        np.savetxt(sol_dir / f"sol_{j:04d}.txt", rng.standard_normal(n))

    out_dir = tmp_path / "ds_archive"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "A_*.txt"),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": n_solutions},
                seed=0,
                shuffle=False,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": str(sol_dir / "sol_*.txt"),
                        "samples": n_solutions,
                    }
                },
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (n_solutions, n)
    assert solutions.shape == (n_solutions, n)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_path), mode="r")
    assert mat_arr.shape[0] == n_solutions
    assert load_matrix_sample_index(out_dir).tolist() == [0, 0, 1, 1, 2]


@pytest.fixture
def subdomain_named_txt_matrices(
    tmp_path: Path, subdomain_matrices: dict[str, np.ndarray]
) -> tuple[Path, int]:
    """The subdomain matrices written to .txt files, for building a real dataset."""
    mat_dir = tmp_path / "subdomain_matrices"
    mat_dir.mkdir()
    for name, matrix in subdomain_matrices.items():
        np.savetxt(mat_dir / name, matrix)
    return mat_dir, 4


def test_build_dataset_honors_enumerate_by_for_globbed_matrix_files(
    subdomain_named_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    mat_dir, n = subdomain_named_txt_matrices
    out_dir = tmp_path / "enumerated_dataset"

    build_test_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "*_subdomain_1_Kaa.txt"),
            enumerate_by=EnumerateBy.NAME,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 3},
                seed=42,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (3, n)
    assert solutions.shape == (3, n)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_path), mode="r")
    assert mat_arr.shape[0] == 3


def test_build_dataset_exclude_indices_removes_matrix_from_generated_dataset(
    subdomain_named_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """A held-out matrix id must never appear in the generated dataset's rows."""
    mat_dir, _n = subdomain_named_txt_matrices
    out_dir = tmp_path / "train_dataset"

    build_test_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "*_subdomain_1_Kaa.txt"),
            enumerate_by=EnumerateBy.NAME,
            exclude_indices=(1,),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 9},
                seed=0,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    matrix_sample_index = load_matrix_sample_index(out_dir)
    assert 1 not in set(matrix_sample_index.tolist())
    assert set(matrix_sample_index.tolist()) == {0, 2}


def test_build_dataset_exclude_indices_keeps_matrix_and_parameters_streams_consistent(
    tmp_path: Path,
) -> None:
    """exclude_indices must apply uniformly to matrix_path and parameters_paths.

    Regression guard: if a future change threads exclude_indices to only one of the
    glob-based streams opened from the same [source] block, bind_sources' strict
    id-matching would raise "parameters IDs must match matrix IDs" for multi-matrix
    sources instead of silently succeeding.
    """
    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir()
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    rng = np.random.default_rng(11)
    parameter_sets = [
        (10554158, 20662154, 19907116, 20715669),
        (10846921, 25611823, 31377296, 28205425),
        (12000001, 22000002, 32000003, 42000004),
    ]
    n = 4
    for e1, e2, e3, e4 in parameter_sets:
        A = rng.standard_normal((n, n))
        A = A @ A.T + n * np.eye(n)
        stem = f"E1_{e1}_E2_{e2}_E3_{e3}_E4_{e4}"
        np.savetxt(mat_dir / f"{stem}_subdomain_1_Kaa.txt", A)
        np.savetxt(param_dir / f"{stem}_YoungModuli_E1_E2_E3_E4.txt", [e1, e2, e3, e4])

    out_dir = tmp_path / "ds_with_params"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "*_subdomain_1_Kaa.txt"),
            parameters_paths=(str(param_dir / "*_YoungModuli_E1_E2_E3_E4.txt"),),
            enumerate_by=EnumerateBy.NAME,
            exclude_indices=(1,),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 3},
                seed=0,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    matrix_sample_index = load_matrix_sample_index(out_dir)
    assert 1 not in set(matrix_sample_index.tolist())
    (parameters,) = load_parameter_arrays(out_dir)
    # Every row's parameters must match its own binding's matrix id (0 or 2), never the
    # excluded matrix's (1) — proves parameters_paths was filtered in lockstep with matrix_path.
    raw_params_by_matrix_id = {0: parameter_sets[0], 2: parameter_sets[2]}
    for row, matrix_id in enumerate(matrix_sample_index.tolist()):
        np.testing.assert_allclose(parameters[row], raw_params_by_matrix_id[matrix_id])


def test_build_dataset_include_indices_builds_holdout_dataset(
    subdomain_named_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """A 1-row-per-matrix holdout dataset exposes exactly the included matrix ids."""
    mat_dir, n = subdomain_named_txt_matrices
    sol_dir = tmp_path / "solutions"
    sol_dir.mkdir()
    rng = np.random.default_rng(0)
    for j in range(2):
        np.savetxt(sol_dir / f"sol_{j:03d}.txt", rng.standard_normal(n))
    out_dir = tmp_path / "holdout_dataset"

    build_test_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "*_subdomain_1_Kaa.txt"),
            enumerate_by=EnumerateBy.NAME,
            include_indices=(0, 2),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": 2},
                seed=0,
                shuffle=False,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": str(sol_dir / "sol_*.txt"),
                        "samples": 2,
                    }
                },
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="zarr",
    )

    matrix_sample_index = load_matrix_sample_index(out_dir)
    assert set(matrix_sample_index.tolist()) == {0, 2}
    rhs, _solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape[0] == 2
    # Distinct rows load distinct physical matrices.
    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_path), mode="r")
    assert not np.allclose(mat_arr[0], mat_arr[1])
