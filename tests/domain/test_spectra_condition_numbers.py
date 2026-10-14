"""Boundary tests for condition-number diagnostic dispatch.

Numerical accuracy belongs to torchalg and SciPy. These tests cover only
neuralls-owned selection, argument forwarding, fallback, and failure mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from scipy.sparse.linalg import ArpackNoConvergence

import neuralls.domain.analysis.spectra as spectra_module
from neuralls.domain.analysis.spectra import compute_condition_numbers


@pytest.fixture
def diagonal_matrix() -> np.ndarray:
    """Small finite matrix for dispatch tests."""
    return np.diag([1.0, 10.0, 100.0])


def _identity(vector: torch.Tensor) -> torch.Tensor:
    return vector


def test_small_matrix_delegates_to_torchalg(
    monkeypatch: pytest.MonkeyPatch,
    diagonal_matrix: np.ndarray,
) -> None:
    calls: list[tuple[torch.Tensor, object]] = []

    def fake_condition_number(matrix: torch.Tensor, preconditioner: object) -> float:
        calls.append((matrix, preconditioner))
        return 12.5

    monkeypatch.setattr(spectra_module, "preconditioned_condition_number", fake_condition_number)

    result = compute_condition_numbers(diagonal_matrix, {"identity": _identity})

    assert result == {"identity": 12.5}
    assert len(calls) == 1
    assert calls[0][0].dtype == torch.float64
    assert calls[0][1] is _identity


def test_large_matrix_delegates_to_arnoldi(
    monkeypatch: pytest.MonkeyPatch,
    diagonal_matrix: np.ndarray,
) -> None:
    monkeypatch.setattr(spectra_module, "_DENSE_FALLBACK_MAX_DIMENSION", 2)
    calls: list[tuple[np.ndarray, torch.Tensor, object]] = []

    def fake_arnoldi(
        matrix: np.ndarray,
        matrix_tensor: torch.Tensor,
        preconditioner: object,
    ) -> float:
        calls.append((matrix, matrix_tensor, preconditioner))
        return 8.0

    monkeypatch.setattr(spectra_module, "_arnoldi_condition_number", fake_arnoldi)

    result = compute_condition_numbers(diagonal_matrix, {"identity": _identity})

    assert result == {"identity": 8.0}
    assert len(calls) == 1
    assert calls[0][0] is diagonal_matrix
    assert calls[0][1].dtype == torch.float64
    assert calls[0][2] is _identity


def test_arnoldi_forwards_bounded_solver_arguments(
    monkeypatch: pytest.MonkeyPatch,
    diagonal_matrix: np.ndarray,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_eigs(*args: object, **kwargs: Any) -> np.ndarray:
        del args
        calls.append(kwargs)
        value = 100.0 if kwargs["which"] == "LM" else 1.0
        return np.array([value])

    monkeypatch.setattr(spectra_module, "eigs", fake_eigs)
    matrix_tensor = torch.as_tensor(diagonal_matrix, dtype=torch.float64)

    result = spectra_module._arnoldi_condition_number(diagonal_matrix, matrix_tensor, _identity)

    assert result == 100.0
    assert [call["which"] for call in calls] == ["LM", "SM"]
    assert all(call["ncv"] == 2 for call in calls)
    assert all(call["maxiter"] == spectra_module._ARPACK_MAXITER for call in calls)
    assert all(call["return_eigenvectors"] is False for call in calls)


def test_arpack_non_convergence_delegates_to_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
    diagonal_matrix: np.ndarray,
) -> None:
    def raising_eigs(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ArpackNoConvergence("forced", np.array([]), np.array([]))

    monkeypatch.setattr(spectra_module, "eigs", raising_eigs)
    monkeypatch.setattr(spectra_module, "preconditioned_condition_number", lambda *_: 7.0)
    matrix_tensor = torch.as_tensor(diagonal_matrix, dtype=torch.float64)

    result = spectra_module._arnoldi_condition_number(diagonal_matrix, matrix_tensor, _identity)

    assert result == 7.0


def test_non_finite_preconditioner_output_maps_to_nan(diagonal_matrix: np.ndarray) -> None:
    def broken_preconditioner(vector: torch.Tensor) -> torch.Tensor:
        return vector * float("nan")

    result = compute_condition_numbers(diagonal_matrix, {"broken": broken_preconditioner})

    assert np.isnan(result["broken"])
