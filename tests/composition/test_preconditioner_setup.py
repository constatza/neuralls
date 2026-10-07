"""Tests for _preconditioner_setup helpers.

Includes regression tests for two bugs:
1. ``_collect_extra_name_positions`` must preserve declaration order, not sort.
2. ``_load_and_bind_extra_inputs`` must bind the correct parameter file to
   each name when the declared names are in non-alphabetical order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torchalg.preconditioners.base import BindableInputs, Preconditioner

from neuralls.composition.comparison._preconditioner_setup import (
    _collect_extra_name_positions,
    _load_and_bind_extra_inputs,
)
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX


class _StubBindable(Preconditioner, BindableInputs):
    """Minimal preconditioner stub that declares extra inputs in a fixed order."""

    def __init__(self, extra_names: tuple[str, ...]) -> None:
        self._names = extra_names
        self.bound_inputs: dict[str, torch.Tensor] = {}

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        return self._names

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        self.bound_inputs.update(inputs)

    def setup(self, matrix: torch.Tensor, context: object | None = None) -> _StubBindable:
        self._mark_ready()
        return self

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        return residual


# ---------------------------------------------------------------------------
# Helpers for building minimal npy-backed datasets with parameter files
# ---------------------------------------------------------------------------

# 2×2 SPD matrix used as a stable fixture across all dataset-backed tests.
_MATRIX: np.ndarray = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
_SAMPLE_INDEX: int = 0

# Distinct parameter arrays: all-ones at position 0 (zebra), all-twos at
# position 1 (apple).  If _load_and_bind_extra_inputs sorts names before
# resolving positions, apple would receive the all-ones array and zebra would
# receive the all-twos array — the opposite of the declaration.
_PARAMS_ZEBRA: np.ndarray = np.ones((1, 2), dtype=np.float64)  # position 0
_PARAMS_APPLE: np.ndarray = np.full((1, 2), 2.0, dtype=np.float64)  # position 1


def _write_npy_dataset_with_params(
    dataset_dir: Path,
    params: tuple[np.ndarray, ...],
) -> None:
    """Write a minimal npy-backed dataset with the given parameter arrays.

    Writes ``matrix.npy``, ``rhs.npy``, ``solutions.npy``, one
    ``parameters_N.npy`` file per entry in *params*, and the manifest.

    Args:
        dataset_dir: Target directory (created if absent).
        params: Sequence of parameter arrays to write at positions 0, 1, ….
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 1
    vec_dim = 2

    matrix_arr = _MATRIX[np.newaxis]  # shape (1, 2, 2)
    rhs_arr = np.array([[1.0, 2.0]], dtype=np.float64)
    solutions_arr = np.array([[0.5, 1.0]], dtype=np.float64)

    np.save(dataset_dir / "matrix.npy", matrix_arr)
    np.save(dataset_dir / "rhs.npy", rhs_arr)
    np.save(dataset_dir / "solutions.npy", solutions_arr)

    params_artifacts: list[DatasetArtifact] = []
    for index, arr in enumerate(params):
        filename = f"{PARAMETERS_ZARR_PREFIX}{index}.npy"
        np.save(dataset_dir / filename, arr)
        params_artifacts.append(
            DatasetArtifact(
                path=filename,
                format="npy",
                dtype="float64",
                shape=tuple(int(d) for d in arr.shape),
                index=index,
            )
        )

    save_dataset_manifest(
        dataset_dir,
        make_dataset_manifest(
            matrix=DatasetArtifact(
                path="matrix.npy",
                format="npy",
                dtype="float64",
                shape=(n_samples, vec_dim, vec_dim),
                n_matrix_samples=n_samples,
                broadcast=False,
            ),
            rhs=DatasetArtifact(
                path="rhs.npy",
                format="npy",
                dtype="float64",
                shape=(n_samples, vec_dim),
            ),
            solutions=DatasetArtifact(
                path="solutions.npy",
                format="npy",
                dtype="float64",
                shape=(n_samples, vec_dim),
            ),
            normalization=DatasetNormalization(
                type="none",
                matrix_norm=1.0,
                matrix_norm_type="spectral",
                scale={},
            ),
            params=tuple(params_artifacts),
        ),
    )


@pytest.fixture
def two_extras_non_alpha() -> dict[str, Preconditioner]:
    return {"model": _StubBindable(("zebra", "apple"))}


@pytest.fixture
def two_extras_alpha() -> dict[str, Preconditioner]:
    return {"model": _StubBindable(("apple", "zebra"))}


def test_collect_extra_name_positions_preserves_declaration_order(
    two_extras_non_alpha: dict[str, Preconditioner],
) -> None:
    positions = _collect_extra_name_positions(two_extras_non_alpha)
    assert positions["zebra"] == 0
    assert positions["apple"] == 1


def test_collect_extra_name_positions_not_alphabetical(
    two_extras_non_alpha: dict[str, Preconditioner],
) -> None:
    positions = _collect_extra_name_positions(two_extras_non_alpha)
    # alphabetical sort would give apple=0, zebra=1 — that must NOT happen
    assert positions["apple"] != 0


def test_collect_extra_name_positions_alpha_declaration_also_correct(
    two_extras_alpha: dict[str, Preconditioner],
) -> None:
    positions = _collect_extra_name_positions(two_extras_alpha)
    assert positions["apple"] == 0
    assert positions["zebra"] == 1


def test_collect_extra_name_positions_deduplicates_across_preconditioners() -> None:
    preconditioners: dict[str, Preconditioner] = {
        "a": _StubBindable(("beta", "alpha")),
        "b": _StubBindable(("beta", "gamma")),
    }
    positions = _collect_extra_name_positions(preconditioners)
    assert positions["beta"] == 0  # first occurrence wins
    assert positions["alpha"] == 1
    assert positions["gamma"] == 1  # position from second preconditioner


def test_collect_extra_name_positions_excludes_matrix() -> None:
    preconditioners: dict[str, Preconditioner] = {"a": _StubBindable(("matrix", "extra"))}
    positions = _collect_extra_name_positions(preconditioners)
    assert "matrix" not in positions
    assert positions["extra"] == 1


# ---------------------------------------------------------------------------
# Fixtures for _load_and_bind_extra_inputs regression (Bug 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_with_two_params(tmp_path: Path) -> Path:
    """Npy-backed dataset dir with two distinct parameter files.

    ``parameters_0.npy`` holds all-ones (2D vector) — mapped to "zebra".
    ``parameters_1.npy`` holds all-twos (2D vector) — mapped to "apple".

    Returns:
        Path to the dataset directory.
    """
    ds_dir = tmp_path / "dataset_two_params"
    _write_npy_dataset_with_params(ds_dir, (_PARAMS_ZEBRA, _PARAMS_APPLE))
    return ds_dir


@pytest.fixture
def stub_bindable_non_alpha() -> _StubBindable:
    """Bindable preconditioner stub with non-alphabetical extra names.

    Declares ``extra_input_names = ("zebra", "apple")``, matching position 0
    and position 1 in the dataset respectively.

    Returns:
        Preconditioner stub that records bound inputs.
    """
    return _StubBindable(("zebra", "apple"))


# ---------------------------------------------------------------------------
# Regression tests for _load_and_bind_extra_inputs (Bug 2)
# ---------------------------------------------------------------------------


def test_load_and_bind_binds_zebra_from_position_0(
    stub_bindable_non_alpha: _StubBindable,
    dataset_with_two_params: Path,
) -> None:
    """_load_and_bind_extra_inputs must bind position-0 data to "zebra".

    Regression: the old sorted-index approach would have enumerated
    ``sorted({"zebra", "apple"}) → ["apple", "zebra"]`` and assigned
    apple→position 0 (wrong) and zebra→position 1 (wrong).
    """
    matrix = torch.as_tensor(_MATRIX, dtype=torch.float64)
    preconditioners: dict[str, Preconditioner] = {"model": stub_bindable_non_alpha}

    _load_and_bind_extra_inputs(
        preconditioners,
        matrix,
        dataset_with_two_params,
        _SAMPLE_INDEX,
    )

    assert "zebra" in stub_bindable_non_alpha.bound_inputs, (
        "'zebra' was not bound at all — positions were not resolved."
    )
    np.testing.assert_allclose(
        stub_bindable_non_alpha.bound_inputs["zebra"].detach().cpu().numpy(),
        _PARAMS_ZEBRA[_SAMPLE_INDEX],
        err_msg=(
            "'zebra' received the wrong parameter array. "
            "Expected all-ones (position 0); the bug would give all-twos (position 1)."
        ),
    )


def test_load_and_bind_binds_apple_from_position_1(
    stub_bindable_non_alpha: _StubBindable,
    dataset_with_two_params: Path,
) -> None:
    """_load_and_bind_extra_inputs must bind position-1 data to "apple"."""
    matrix = torch.as_tensor(_MATRIX, dtype=torch.float64)
    preconditioners: dict[str, Preconditioner] = {"model": stub_bindable_non_alpha}

    _load_and_bind_extra_inputs(
        preconditioners,
        matrix,
        dataset_with_two_params,
        _SAMPLE_INDEX,
    )

    assert "apple" in stub_bindable_non_alpha.bound_inputs, "'apple' was not bound at all."
    np.testing.assert_allclose(
        stub_bindable_non_alpha.bound_inputs["apple"].detach().cpu().numpy(),
        _PARAMS_APPLE[_SAMPLE_INDEX],
        err_msg=(
            "'apple' received the wrong parameter array. "
            "Expected all-twos (position 1); the bug would give all-ones (position 0)."
        ),
    )


def test_load_and_bind_does_not_swap_arrays_for_non_alpha_names(
    stub_bindable_non_alpha: _StubBindable,
    dataset_with_two_params: Path,
) -> None:
    """The arrays bound to zebra and apple must not be swapped.

    This single assertion captures the failure mode of the original bug:
    arrays swapped because sorted() alphabetized the names before assigning
    positional indices.
    """
    matrix = torch.as_tensor(_MATRIX, dtype=torch.float64)
    preconditioners: dict[str, Preconditioner] = {"model": stub_bindable_non_alpha}

    _load_and_bind_extra_inputs(
        preconditioners,
        matrix,
        dataset_with_two_params,
        _SAMPLE_INDEX,
    )

    bound = stub_bindable_non_alpha.bound_inputs
    zebra_val = float(bound["zebra"][0])
    apple_val = float(bound["apple"][0])

    assert zebra_val == pytest.approx(1.0), (
        f"'zebra' bound value should be 1.0 (all-ones), got {zebra_val}. "
        "Indicates the arrays were swapped."
    )
    assert apple_val == pytest.approx(2.0), (
        f"'apple' bound value should be 2.0 (all-twos), got {apple_val}. "
        "Indicates the arrays were swapped."
    )
