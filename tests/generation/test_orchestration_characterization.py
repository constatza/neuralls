"""Byte-exact characterization of the generation orchestration call chain.

These tests pin the *observable output* of the full
``build_dataset -> build_dataset_payload -> _prepare_generation_context ->
_open_streams / _resolve_binding_strategy_counts -> _accumulate_bindings ->
_process_binding -> _generate_mixture_with_metadata -> _finalize_payload``
chain for fixed seeds, so parameter-threading refactors of those functions can
be proven behavior-preserving rather than merely test-passing.

Each expected digest was captured from the pre-refactor implementation. A digest
change means the generated dataset changed — that is a regression, not a test
that needs updating.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.composition.generation.process_data import process_data_from_config
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
)


def _digest(array: np.ndarray) -> str:
    """Stable content digest of an array's dtype, shape and exact bytes."""
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()[:16]


@pytest.fixture
def spd_matrix_dir(tmp_path: Path) -> Path:
    """Three deterministic 5x5 SPD matrices as individually globbable .txt files."""
    rng = np.random.default_rng(11)
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    for index in range(3):
        base = rng.standard_normal((5, 5))
        np.savetxt(matrix_dir / f"A_{index:03d}.txt", base @ base.T + 5.0 * np.eye(5))
    return matrix_dir


@pytest.fixture
def single_spd_matrix(tmp_path: Path) -> Path:
    """One deterministic 4x4 SPD matrix as a .npy file."""
    rng = np.random.default_rng(5)
    base = rng.standard_normal((4, 4))
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, base @ base.T + 4.0 * np.eye(4))
    return matrix_path


@pytest.fixture
def solution_archive_dir(tmp_path: Path) -> Path:
    """Five deterministic 5-vectors forming a globbable solution archive."""
    archive_dir = tmp_path / "solutions"
    archive_dir.mkdir()
    for index in range(5):
        np.savetxt(archive_dir / f"x_{index:03d}.txt", np.full(5, float(index + 1)))
    return archive_dir


@pytest.fixture
def parameter_files(tmp_path: Path) -> tuple[str, ...]:
    """One per-sample parameter vector bound to a single-matrix source."""
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    np.savetxt(params_dir / "p_000.txt", np.array([0.25, 0.5, 0.75]))
    return (str(params_dir / "p_*.txt"),)


def test_multi_matrix_synthetic_mixture_output_is_stable(
    spd_matrix_dir: Path, tmp_path: Path, solver_overrides: dict[str, Any]
) -> None:
    """Multi-matrix synthetic mixture: allocation, row kinds and matrix index are pinned.

    Exercises _resolve_binding_strategy_counts (multi-binding allocation),
    _accumulate_bindings (MANY_MATRICES accumulator path), _process_binding and
    _generate_mixture_with_metadata's trace-strategy flattening.
    """
    out_dir = tmp_path / "dataset"
    build_dataset(
        SourceSpec(
            matrix_path=str(spd_matrix_dir / "A_*.txt"),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 6, "gaussian_residuals": 6},
                seed=1234,
                shuffle=False,
                strategy_overrides={"gaussian_residuals": {"cg_iters": 2}},
                solver_overrides=solver_overrides,
            ),
            normalize="matrix",
        ),
        str(out_dir),
        dataset_format="npy",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    manifest = load_dataset_manifest(out_dir)

    assert (rhs.shape, solutions.shape) == ((12, 5), (12, 5))
    assert _digest(rhs) == "8b7d7df504b2b5cb"
    assert _digest(solutions) == "890c1438c4ab7872"
    assert _digest(load_row_kind_codes(out_dir)) == "c79d2a9c87907f68"
    assert _digest(load_matrix_sample_index(out_dir)) == "ef34a7ad43e00545"
    assert _digest(load_matrix_dense_sample(out_dir, 0)) == "fc1a0250f33915bc"
    assert manifest["matrix"]["shape"] == [12, 5, 5]
    assert manifest["normalization"]["type"] == "matrix"
    assert manifest["normalization"]["matrix_norm"] == pytest.approx(0.36070429238817936)
    assert manifest["normalization"]["scale"] == {}


def test_multi_matrix_solution_archive_all_samples_output_is_stable(
    spd_matrix_dir: Path, solution_archive_dir: Path, tmp_path: Path
) -> None:
    """samples=-1 across bindings: archive-size resolution and disjoint skips are pinned.

    Exercises _resolve_all_samples_total, the per-binding cumulative skip offsets
    and _merge_binding_skip_overrides. The matrix_sample_index [0,0,1,1,2] is the
    load-bearing assertion: it proves each binding drew its own disjoint slice of
    the 5-file archive (2/2/1) instead of every binding reloading all five.
    """
    out_dir = tmp_path / "dataset"
    build_dataset(
        SourceSpec(
            matrix_path=str(spd_matrix_dir / "A_*.txt"),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": -1},
                seed=99,
                shuffle=False,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": str(solution_archive_dir / "x_*.txt"),
                        "shuffle": False,
                    }
                },
            ),
            normalize="none",
        ),
        str(out_dir),
        dataset_format="npy",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)

    assert (rhs.shape, solutions.shape) == ((5, 5), (5, 5))
    assert _digest(rhs) == "54e08358e2284a45"
    assert _digest(solutions) == "e9b5432441f9d566"
    assert load_matrix_sample_index(out_dir).tolist() == [0, 0, 1, 1, 2]
    # Each archive vector is the constant vector of its 1-based file index; the
    # whole archive is consumed exactly once, in order, across the three bindings.
    np.testing.assert_array_equal(
        solutions, np.vstack([np.full(5, float(i + 1)) for i in range(5)])
    )


def test_single_matrix_with_parameter_streams_output_is_stable(
    single_spd_matrix: Path, parameter_files: tuple[str, ...], tmp_path: Path
) -> None:
    """Single-matrix broadcast layout with parameter streams is pinned.

    Exercises _open_streams' parameter-stream binding, the BROADCAST_SINGLE
    accumulator branch in _accumulate_bindings (one matrix written, not one per
    sample) and the param_blocks stacking in _finalize_payload.
    """
    out_dir = tmp_path / "dataset"
    build_dataset(
        SourceSpec(
            matrix_path=str(single_spd_matrix),
            parameters_paths=parameter_files,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 4},
                seed=7,
                shuffle=True,
            ),
            normalize="matrix",
        ),
        str(out_dir),
        dataset_format="npy",
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    parameters = load_parameter_arrays(out_dir)

    assert (rhs.shape, solutions.shape) == ((4, 4), (4, 4))
    assert _digest(rhs) == "3db211f8b2da4370"
    assert _digest(solutions) == "075009b30984eba4"
    assert len(parameters) == 1
    assert parameters[0].shape == (4, 3)
    assert _digest(parameters[0]) == "d035f298199f613c"
    assert load_dataset_manifest(out_dir)["matrix"]["shape"] == [1, 4, 4]


def _write_config(tmp_path: Path, dataset_id: str, body: str) -> Path:
    """Write a data config TOML whose id matches the requested dataset id."""
    config_path = tmp_path / f"{dataset_id}.toml"
    config_path.write_text(f'id = "{dataset_id}"\n{body}\n[output]\n')
    return config_path


@pytest.fixture
def identity_matrix_file(tmp_path: Path) -> Path:
    """A 2x2 matrix of 2*I, so solutions are exactly half of each RHS."""
    matrix_path = tmp_path / "matrix.txt"
    np.savetxt(matrix_path, 2.0 * np.eye(2))
    return matrix_path


@pytest.fixture
def rhs_archive_dir(tmp_path: Path) -> Path:
    """Three RHS vectors forming a globbable RHS archive."""
    archive_dir = tmp_path / "rhs"
    archive_dir.mkdir()
    for index in range(3):
        np.savetxt(archive_dir / f"b_{index:03d}.txt", np.array([index + 1.0, index + 2.0]))
    return archive_dir


def test_rhs_archive_only_executor_produces_solved_pairs(
    tmp_path: Path, identity_matrix_file: Path, rhs_archive_dir: Path, neuralls_settings: Any
) -> None:
    """The rhs-archive-only executor branch solves each archived RHS against the matrix.

    This is the one _execute_plan branch with no other end-to-end coverage; it
    pins that `source.rhs_path` reaches the strategy as an archive glob (not as
    a per-binding RHS stream) and that A=2I yields solutions of exactly rhs/2.
    """
    config_path = _write_config(
        tmp_path,
        "rhs-archive-dataset",
        f"""
[source]
matrix_path = "{identity_matrix_file.as_posix()}"
rhs_path = "{(rhs_archive_dir / "b_*.txt").as_posix()}"

[generation]
normalize = "none"
shuffle = false

[[generation.strategy]]
name = "rhs_archive"
samples = -1
""",
    )

    output_dir = process_data_from_config(config_path, neuralls_settings)

    rhs, solutions = load_dense_training_arrays(output_dir)
    assert rhs.shape == (3, 2)
    np.testing.assert_allclose(np.sort(rhs, axis=0), [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(solutions, rhs / 2.0)


def test_solution_archive_only_executor_persists_archive_vectors(
    tmp_path: Path, identity_matrix_file: Path, neuralls_settings: Any
) -> None:
    """The solution-archive-only executor branch persists the archive as solutions.

    Existing coverage of this branch only asserted that the output directory
    exists; this pins the actual arrays, so a regression in the executor's
    source/spec threading is caught rather than silently producing a directory.
    """
    solutions_dir = tmp_path / "solutions"
    solutions_dir.mkdir()
    for index in range(3):
        np.savetxt(solutions_dir / f"x_{index:03d}.txt", np.array([index + 1.0, index + 2.0]))

    config_path = _write_config(
        tmp_path,
        "solution-archive-dataset",
        f"""
[source]
matrix_path = "{identity_matrix_file.as_posix()}"
solutions_path = "{(solutions_dir / "x_*.txt").as_posix()}"

[generation]
normalize = "none"
shuffle = false

[[generation.strategy]]
name = "solution_archive"
samples = -1
""",
    )

    output_dir = process_data_from_config(config_path, neuralls_settings)

    rhs, solutions = load_dense_training_arrays(output_dir)
    assert solutions.shape == (3, 2)
    np.testing.assert_allclose(np.sort(solutions, axis=0), [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    # A = 2I, so the recorded RHS must be exactly A @ x for every stored solution.
    np.testing.assert_allclose(rhs, solutions * 2.0)
