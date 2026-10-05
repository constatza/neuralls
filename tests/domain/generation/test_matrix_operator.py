"""Tests for MatrixOperator, the solve registry, and the sparse eigen rule."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import csr_array

from neuralls.domain.generation.helpers import (
    _compute_eigendecomposition,
    _select_eigenvectors,
    _solve_linear_systems,
)
from neuralls.domain.generation.matrix_operator import MatrixOperator
from neuralls.domain.generation.strategy_configs import require_which_supported_by_format
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.shared.types import MatrixFormat

FIXTURE_SEED = 7
MATRIX_SIZE = 12
SPARSE_DENSITY = 0.3
EIGEN_COUNT = 3
DIRECT_TOLERANCE = 1e-10
CG_TOLERANCE = 1e-6
EIGEN_TOLERANCE = 1e-8
RHS_COUNT = 4


class _NoDensifyCsr(csr_array):
    """CSR array whose densifying methods fail, to prove a path never densifies."""

    def toarray(self, order: str | None = None, out: np.ndarray | None = None) -> np.ndarray:
        raise AssertionError("toarray called on the CSR path")

    def todense(self, order: str | None = None, out: np.ndarray | None = None) -> np.ndarray:
        raise AssertionError("todense called on the CSR path")


def _spd_source(seed: int, n: int) -> np.ndarray:
    """Seeded sparse-pattern symmetric matrix shifted to be SPD."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < SPARSE_DENSITY
    mask = mask | mask.T
    values = rng.standard_normal((n, n)) * mask
    symmetric = (values + values.T) / 2.0
    return symmetric + n * np.eye(n)


def _rhs_rows(seed: int, n: int) -> np.ndarray:
    """Seeded right-hand side rows, shape (RHS_COUNT, n)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((RHS_COUNT, n))


@pytest.fixture
def spd_dense() -> np.ndarray:
    """Dense seeded SPD matrix."""
    return _spd_source(FIXTURE_SEED, MATRIX_SIZE)


@pytest.fixture
def spd_csr(spd_dense: np.ndarray) -> _NoDensifyCsr:
    """CSR form of the same SPD matrix, guarded against densification."""
    return _NoDensifyCsr(csr_array(spd_dense))


@pytest.fixture
def rhs_rows() -> np.ndarray:
    """Seeded RHS rows matching the SPD fixtures."""
    return _rhs_rows(FIXTURE_SEED + 1, MATRIX_SIZE)


def test_format_is_derived_from_matrix_type(spd_dense: np.ndarray, spd_csr: _NoDensifyCsr) -> None:
    """The operator reports the format of the matrix it wraps."""
    assert MatrixOperator(spd_dense).format is MatrixFormat.DENSE
    assert MatrixOperator(spd_csr).format is MatrixFormat.CSR


def test_non_matrix_input_is_rejected() -> None:
    """A scipy LinearOperator has no factorization and must be rejected."""
    from scipy.sparse.linalg import aslinearoperator

    with pytest.raises(TypeError, match="requires a dense ndarray or csr_array"):
        MatrixOperator(aslinearoperator(np.eye(3)))  # type: ignore[arg-type]


@pytest.mark.parametrize("matrix_kind", ["dense", "csr"])
def test_solve_direct_matches_numpy(
    matrix_kind: str,
    spd_dense: np.ndarray,
    spd_csr: _NoDensifyCsr,
    rhs_rows: np.ndarray,
) -> None:
    """Direct solve agrees with np.linalg.solve for both formats."""
    matrix = spd_dense if matrix_kind == "dense" else spd_csr
    operator = MatrixOperator(matrix)
    for rhs in rhs_rows:
        expected = np.linalg.solve(spd_dense, rhs)
        np.testing.assert_allclose(operator.solve_direct(rhs), expected, atol=DIRECT_TOLERANCE)


def test_factor_is_built_once_and_reused(spd_dense: np.ndarray, rhs_rows: np.ndarray) -> None:
    """Repeated solves on one operator reuse the cached factorization."""
    operator = MatrixOperator(spd_dense)
    cache = operator._cache
    assert cache.factor is None

    operator.solve_direct(rhs_rows[0])
    first_factor = cache.factor
    assert first_factor is not None

    operator.solve_direct(rhs_rows[1])
    operator.solve_direct(rhs_rows[2])
    assert operator._cache is cache
    assert cache.factor is first_factor


@pytest.mark.parametrize("which", ["smallest", "largest"])
def test_csr_eigensystem_matches_dense_eigh(
    which: str, spd_dense: np.ndarray, spd_csr: _NoDensifyCsr
) -> None:
    """Sparse eigensolve returns the same eigenvalues as dense eigh at each end."""
    all_values = eigh(spd_dense, eigvals_only=True)
    expected = all_values[:EIGEN_COUNT] if which == "smallest" else all_values[-EIGEN_COUNT:]

    values, vectors = MatrixOperator(spd_csr).eigensystem(EIGEN_COUNT, which)  # type: ignore[arg-type]

    np.testing.assert_allclose(values, expected, atol=EIGEN_TOLERANCE)
    assert vectors.shape == (MATRIX_SIZE, EIGEN_COUNT)


def test_csr_eigensystem_rejects_full_spectrum(spd_csr: _NoDensifyCsr) -> None:
    """ARPACK cannot return the full spectrum; the error points at dense."""
    with pytest.raises(ValueError, match="use a dense matrix"):
        MatrixOperator(spd_csr).eigensystem(MATRIX_SIZE, "smallest")


def test_eigensystem_rejects_asymmetric_matrix() -> None:
    """Asymmetric input is rejected before any eigen computation."""
    asymmetric = np.triu(np.ones((4, 4)))
    with pytest.raises(ValueError, match="require symmetric matrices"):
        MatrixOperator(asymmetric).eigensystem(2, "smallest")


@pytest.mark.parametrize("method", ["direct", "cg"])
def test_registry_solvers_agree_across_formats(
    method: str,
    spd_dense: np.ndarray,
    spd_csr: _NoDensifyCsr,
    rhs_rows: np.ndarray,
) -> None:
    """DIRECT and CG give the same solutions for dense and CSR operators."""
    tolerance = DIRECT_TOLERANCE if method == "direct" else CG_TOLERANCE
    expected = np.array([np.linalg.solve(spd_dense, rhs) for rhs in rhs_rows])

    dense_solutions = _solve_linear_systems(MatrixOperator(spd_dense), rhs_rows, method)  # type: ignore[arg-type]
    csr_solutions = _solve_linear_systems(MatrixOperator(spd_csr), rhs_rows, method)  # type: ignore[arg-type]

    np.testing.assert_allclose(dense_solutions, expected, atol=tolerance)
    np.testing.assert_allclose(csr_solutions, expected, atol=tolerance)


def test_csr_eigen_random_selection_is_rejected(spd_csr: _NoDensifyCsr) -> None:
    """Random selection needs the full spectrum, so CSR must refuse it."""
    with pytest.raises(ValueError, match="requires a dense matrix"):
        _compute_eigendecomposition(MatrixOperator(spd_csr), EIGEN_COUNT, "random")


def test_dense_random_selection_uses_full_spectrum(spd_dense: np.ndarray) -> None:
    """Dense random selection draws from the full eigh spectrum, as before."""
    operator = MatrixOperator(spd_dense)
    eigenvalues, eigenvectors = _compute_eigendecomposition(operator, EIGEN_COUNT, "random")
    expected_values, expected_vectors = eigh(spd_dense)

    np.testing.assert_allclose(eigenvalues, expected_values, atol=EIGEN_TOLERANCE)
    np.testing.assert_allclose(np.abs(eigenvectors), np.abs(expected_vectors), atol=EIGEN_TOLERANCE)

    selected_a, _, indices_a = _select_eigenvectors(
        eigenvectors, eigenvalues, EIGEN_COUNT, "random", np.random.default_rng(FIXTURE_SEED)
    )
    selected_b, _, indices_b = _select_eigenvectors(
        expected_vectors,
        expected_values,
        EIGEN_COUNT,
        "random",
        np.random.default_rng(FIXTURE_SEED),
    )
    np.testing.assert_array_equal(indices_a, indices_b)
    np.testing.assert_allclose(np.abs(selected_a), np.abs(selected_b), atol=EIGEN_TOLERANCE)


def test_config_rejects_random_for_csr_dataset() -> None:
    """The shared format rule rejects random selection for CSR."""
    with pytest.raises(ValueError, match="requires the full spectrum"):
        require_which_supported_by_format("random", MatrixFormat.CSR)


@pytest.mark.parametrize("matrix_format", ["csr", "dense"])
def test_data_config_checks_eigen_which_against_output_format(matrix_format: str) -> None:
    """The data config file rejects random selection only for CSR datasets."""
    raw = {
        "id": "eigen-check",
        "output": {"matrix_format": matrix_format},
        "generation": {
            "strategy": [
                {"name": "eigenvector_forward", "samples": 2, "which": "random"},
            ]
        },
    }
    if matrix_format == "csr":
        with pytest.raises(ValueError, match="requires the full spectrum"):
            DataConfigFile.model_validate(raw)
    else:
        DataConfigFile.model_validate(raw)
