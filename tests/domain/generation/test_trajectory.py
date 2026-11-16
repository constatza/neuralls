"""The shared snapshot-trajectory steps: propagation scale invariance and row scaling."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.composition.generation.default_services import make_solver
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.step_window import StepWindow
from neuralls.domain.generation.strategy_configs import NormSnapshotScaling, NoSnapshotScaling
from neuralls.domain.generation.trajectory import (
    cg_error_traces,
    draw_probes,
    jacobi_probe_traces,
    scale_rows,
)
from neuralls.domain.generation.transforms import ComputeRhsTransform
from neuralls.shared.types import SystemMatrix

_DIMENSION = 10
_ALPHA = 37.5
_WINDOW = StepWindow(stop=5, start=1)
_PROPAGATION_RTOL = 1e-8
_PROPAGATION_ATOL = 1e-10
"""Roundoff tolerances: the auto-omega spectral-radius estimate is invariant only up to
its own rounding (about 1e-9 relative), the rest far below it."""


@pytest.fixture
def dense_spd() -> np.ndarray:
    """A seeded, well-conditioned SPD matrix with a sparse-friendly pattern."""
    base = np.random.default_rng(5).standard_normal((_DIMENSION, _DIMENSION))
    banded = np.triu(np.tril(base @ base.T, 2), -2)
    return banded + np.diag(np.abs(banded).sum(axis=1))


@pytest.fixture(params=["dense", "csr"])
def as_format(request: pytest.FixtureRequest) -> Callable[[np.ndarray], SystemMatrix]:
    """Store a dense matrix in each format the trajectories accept."""
    if request.param == "dense":
        return lambda dense: dense
    return csr_array


@pytest.fixture
def matrix(dense_spd: np.ndarray, as_format: Callable[[np.ndarray], SystemMatrix]) -> SystemMatrix:
    """The SPD matrix A."""
    return as_format(dense_spd)


@pytest.fixture
def scaled_matrix(
    dense_spd: np.ndarray, as_format: Callable[[np.ndarray], SystemMatrix]
) -> SystemMatrix:
    """``A / alpha`` in the same format as ``matrix``."""
    return as_format(dense_spd / _ALPHA)


@pytest.fixture
def probes() -> np.ndarray:
    """Three seeded Gaussian probes."""
    return draw_probes(np.random.default_rng(9), 3, _DIMENSION, "gaussian")


@pytest.fixture
def solver() -> TracingSolverCallable:
    """The production tracing CG solver."""
    return make_solver()


@pytest.fixture
def snapshot_pairs(dense_spd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consistent ``(r, e)`` rows with ``r = A e``, the last row zero."""
    errors = np.random.default_rng(13).standard_normal((4, _DIMENSION))
    errors[-1] = 0.0
    return errors @ dense_spd.T, errors


def test_draw_probes_draws_the_requested_distribution() -> None:
    gaussian = draw_probes(np.random.default_rng(0), 4, 6, "gaussian")
    rademacher = draw_probes(np.random.default_rng(0), 4, 6, "rademacher")

    np.testing.assert_array_equal(gaussian, np.random.default_rng(0).standard_normal((4, 6)))
    assert set(np.unique(rademacher)) == {-1.0, 1.0}


def test_cg_errors_are_invariant_under_matrix_scaling(
    matrix: SystemMatrix,
    scaled_matrix: SystemMatrix,
    probes: np.ndarray,
    solver: TracingSolverCallable,
) -> None:
    """CG iterates do not change when A and b are divided by alpha; residuals scale by 1/alpha."""
    reference = cg_error_traces(
        matrix, ComputeRhsTransform(matrix).transform(probes), probes, _WINDOW, solver
    )
    result = cg_error_traces(
        scaled_matrix, ComputeRhsTransform(scaled_matrix).transform(probes), probes, _WINDOW, solver
    )

    np.testing.assert_allclose(
        result.errors, reference.errors, rtol=_PROPAGATION_RTOL, atol=_PROPAGATION_ATOL
    )
    np.testing.assert_allclose(
        result.residuals * _ALPHA,
        reference.residuals,
        rtol=_PROPAGATION_RTOL,
        atol=_PROPAGATION_ATOL,
    )


@pytest.mark.parametrize("omega", [None, 0.6])
def test_jacobi_probes_are_invariant_under_matrix_scaling(
    matrix: SystemMatrix, scaled_matrix: SystemMatrix, probes: np.ndarray, omega: float | None
) -> None:
    """Weighted-Jacobi sweeps depend on D^-1 A only; residuals A e scale by 1/alpha."""
    reference = jacobi_probe_traces(matrix, probes, _WINDOW, omega)
    result = jacobi_probe_traces(scaled_matrix, probes, _WINDOW, omega)

    np.testing.assert_allclose(
        result.solutions, reference.solutions, rtol=_PROPAGATION_RTOL, atol=_PROPAGATION_ATOL
    )
    np.testing.assert_allclose(
        result.residuals * _ALPHA,
        reference.residuals,
        rtol=_PROPAGATION_RTOL,
        atol=_PROPAGATION_ATOL,
    )


def test_no_snapshot_scaling_returns_the_rows_unchanged(
    snapshot_pairs: tuple[np.ndarray, np.ndarray],
) -> None:
    residuals, errors = snapshot_pairs

    scaled_residuals, scaled_errors = scale_rows(residuals, errors, NoSnapshotScaling())

    assert scaled_residuals is residuals
    assert scaled_errors is errors


@pytest.mark.parametrize("norm", ["l2", "energy"])
def test_power_zero_is_the_identity(
    snapshot_pairs: tuple[np.ndarray, np.ndarray], norm: str
) -> None:
    residuals, errors = snapshot_pairs
    scaling = NormSnapshotScaling.model_validate({"norm": norm, "power": 0.0})

    scaled_residuals, scaled_errors = scale_rows(residuals, errors, scaling)

    np.testing.assert_array_equal(scaled_residuals, residuals)
    np.testing.assert_array_equal(scaled_errors, errors)


def test_power_one_l2_gives_unit_error_rows(
    snapshot_pairs: tuple[np.ndarray, np.ndarray],
) -> None:
    residuals, errors = snapshot_pairs

    _, scaled_errors = scale_rows(residuals, errors, NormSnapshotScaling(norm="l2"))

    np.testing.assert_allclose(np.linalg.norm(scaled_errors[:-1], axis=1), 1.0, rtol=1e-12)


def test_power_one_energy_gives_unit_energy_rows(
    snapshot_pairs: tuple[np.ndarray, np.ndarray], dense_spd: np.ndarray
) -> None:
    residuals, errors = snapshot_pairs

    _, scaled_errors = scale_rows(residuals, errors, NormSnapshotScaling(norm="energy"))

    energy = np.einsum("ij,ij->i", scaled_errors, scaled_errors @ dense_spd.T)
    np.testing.assert_allclose(energy[:-1], 1.0, rtol=1e-12)


@pytest.mark.parametrize("norm", ["l2", "energy"])
def test_scaling_keeps_every_pair_consistent_with_the_matrix(
    snapshot_pairs: tuple[np.ndarray, np.ndarray], dense_spd: np.ndarray, norm: str
) -> None:
    """Both rows of a pair get the same factor, so A e = r still holds."""
    residuals, errors = snapshot_pairs
    scaling = NormSnapshotScaling.model_validate({"norm": norm, "power": 0.5})

    scaled_residuals, scaled_errors = scale_rows(residuals, errors, scaling)

    np.testing.assert_allclose(scaled_residuals, scaled_errors @ dense_spd.T, rtol=1e-12)


@pytest.mark.parametrize("norm", ["l2", "energy"])
def test_zero_rows_are_left_unchanged(
    snapshot_pairs: tuple[np.ndarray, np.ndarray], norm: str
) -> None:
    residuals, errors = snapshot_pairs
    scaling = NormSnapshotScaling.model_validate({"norm": norm})

    scaled_residuals, scaled_errors = scale_rows(residuals, errors, scaling)

    np.testing.assert_array_equal(scaled_residuals[-1], np.zeros(_DIMENSION))
    np.testing.assert_array_equal(scaled_errors[-1], np.zeros(_DIMENSION))


def test_energy_scaling_rejects_a_negative_energy() -> None:
    errors = np.ones((1, 3))

    with pytest.raises(ValueError):
        scale_rows(-errors, errors, NormSnapshotScaling(norm="energy"))
