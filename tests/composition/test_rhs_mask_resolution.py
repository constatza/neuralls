"""A mask zeroes the resolved RHS once, for every source kind, on its own random stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison.models import ResolvedComparisonInput
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind
from tests.dataset_support import build_test_dataset

pytestmark = pytest.mark.deferred_io("generation-io")

DIM = 40
SEED = 17
MASK = {"kind": "random", "density": 0.25}
SUPPORT_SIZE = round(0.25 * DIM)


@pytest.fixture
def matrix_file(tmp_path: Path) -> Path:
    path = tmp_path / "matrix.npy"
    rng = np.random.default_rng(0)
    np.save(path, np.eye(DIM) * 4.0 + 0.01 * rng.standard_normal((DIM, DIM)))
    return path


@pytest.fixture
def solution_file(tmp_path: Path) -> Path:
    path = tmp_path / "solution.npy"
    np.save(path, np.random.default_rng(1).standard_normal(DIM))
    return path


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    matrix_path = tmp_path / "small.npy"
    np.save(matrix_path, np.array([[4.0, -1.0], [-1.0, 3.0]]))
    out = tmp_path / "dataset"
    build_test_dataset(
        SourceSpec(matrix_path=str(matrix_path)),
        DatasetSpec(
            mixture=MixtureSpec(counts={"neutral_ones": 3}, seed=42, shuffle=False),
            normalize="none",
        ),
        str(out),
        dataset_format="hdf5",
    )
    return out


def _resolve(
    matrix_path: Path,
    kind: ComparisonRhsSourceKind,
    params: dict[str, object],
    seed: int | None = SEED,
) -> ResolvedComparisonInput:
    return resolve_comparison_input(
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        matrix_path=matrix_path,
        matrix_dataset_id="ds",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=seed,
        rhs_source_kind=kind,
        rhs_source_params=params,
    )


def _gaussian(matrix_file: Path, mask: dict[str, object] | None, seed: int = SEED):
    params: dict[str, object] = {} if mask is None else {"mask": mask}
    return _resolve(matrix_file, ComparisonRhsSourceKind.GAUSSIAN, params, seed)


def test_a_masked_gaussian_keeps_the_rounded_density_of_entries(matrix_file: Path) -> None:
    resolved = _gaussian(matrix_file, MASK)

    assert np.count_nonzero(resolved.rhs) == SUPPORT_SIZE


def test_the_mask_keeps_the_values_of_the_unmasked_draw_on_its_support(matrix_file: Path) -> None:
    plain = _gaussian(matrix_file, None).rhs
    masked = _gaussian(matrix_file, MASK).rhs
    support = masked != 0.0

    np.testing.assert_array_equal(masked[support], plain[support])


def test_the_mask_stream_is_independent_of_the_rhs_stream(matrix_file: Path) -> None:
    """Changing the density must not change the underlying draw, and the support is reproducible."""
    plain = _gaussian(matrix_file, None).rhs
    sparser = _gaussian(matrix_file, {"kind": "random", "density": 0.1}).rhs
    denser = _gaussian(matrix_file, {"kind": "random", "density": 0.5}).rhs

    for masked in (sparser, denser):
        support = masked != 0.0
        np.testing.assert_array_equal(masked[support], plain[support])
    np.testing.assert_array_equal(
        sparser, _gaussian(matrix_file, {"kind": "random", "density": 0.1}).rhs
    )


def test_systems_with_different_seeds_get_different_supports(matrix_file: Path) -> None:
    first = _gaussian(matrix_file, MASK, seed=1).rhs != 0.0
    second = _gaussian(matrix_file, MASK, seed=2).rhs != 0.0

    assert not np.array_equal(first, second)


def test_a_masked_raw_lhs_system_drops_its_known_solution(
    dataset_dir: Path, solution_file: Path, tmp_path: Path
) -> None:
    small_solution = tmp_path / "small_solution.npy"
    np.save(small_solution, np.array([2.0, -1.0]))
    params: dict[str, object] = {
        "path": small_solution,
        "row_kind": RowKind.STANDARD,
        "mask": {"kind": "random", "density": 0.5},
    }

    unmasked = _resolve(
        dataset_dir, ComparisonRhsSourceKind.RAW_LHS, {**params, "mask": {"kind": "none"}}
    )
    masked = _resolve(dataset_dir, ComparisonRhsSourceKind.RAW_LHS, params)

    assert unmasked.lhs is not None
    assert masked.lhs is None
    assert np.count_nonzero(masked.rhs) == 1


def test_a_masked_dataset_system_drops_its_known_solution(dataset_dir: Path) -> None:
    unmasked = _resolve(dataset_dir, ComparisonRhsSourceKind.DATASET, {"sample_index": 0})
    masked = _resolve(
        dataset_dir,
        ComparisonRhsSourceKind.DATASET,
        {"sample_index": 0, "mask": {"kind": "random", "density": 0.5}},
    )

    assert unmasked.lhs is not None
    assert masked.lhs is None
    assert np.count_nonzero(masked.rhs) == 1
    support = masked.rhs != 0.0
    np.testing.assert_array_equal(masked.rhs[support], unmasked.rhs[support])


def test_an_unmasked_system_keeps_its_known_solution(dataset_dir: Path) -> None:
    resolved = _resolve(dataset_dir, ComparisonRhsSourceKind.DATASET, {"sample_index": 0})

    assert resolved.lhs is not None
