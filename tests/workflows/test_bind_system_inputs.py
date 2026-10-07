"""Tests for _bind_system_inputs helper in comparison run workflow."""

from __future__ import annotations

import pytest
import torch
from torchalg.preconditioners.base import (
    BindableInputs,
    Preconditioner,
    PreconditionerContext,
)

from neuralls.composition.comparison._preconditioner_setup import _bind_system_inputs


class _TrackingPreconditioner(Preconditioner, BindableInputs):
    """Preconditioner that records which extra inputs were bound to it."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names
        self.bound: dict[str, torch.Tensor] = {}

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        return self._names

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        self.bound = dict(inputs)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> _TrackingPreconditioner:
        self._mark_ready()
        return self

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        return residual.clone()


@pytest.fixture
def matrix() -> torch.Tensor:
    """5x5 identity matrix."""
    return torch.eye(5, dtype=torch.float64)


@pytest.fixture
def coords() -> torch.Tensor:
    """5x2 coordinate array."""
    return torch.arange(10.0, dtype=torch.float64).reshape(5, 2)


def test_bind_matrix_to_neural_only(matrix: torch.Tensor) -> None:
    """Only the preconditioner that declares 'matrix' receives the binding."""
    neural = _TrackingPreconditioner(("matrix",))
    jacobi = _TrackingPreconditioner(())
    _bind_system_inputs({"neural": neural, "jacobi": jacobi}, {"matrix": matrix})
    assert torch.equal(neural.bound["matrix"], matrix)
    assert jacobi.bound == {}


def test_bind_multiple_extras(matrix: torch.Tensor, coords: torch.Tensor) -> None:
    """A preconditioner declaring multiple names gets all available extras bound."""
    precond = _TrackingPreconditioner(("matrix", "coordinates"))
    _bind_system_inputs({"p": precond}, {"matrix": matrix, "coordinates": coords})
    assert torch.equal(precond.bound["matrix"], matrix)
    assert torch.equal(precond.bound["coordinates"], coords)


def test_missing_extra_not_bound(matrix: torch.Tensor) -> None:
    """Names declared but absent from system_data are silently omitted."""
    precond = _TrackingPreconditioner(("matrix", "coordinates"))
    _bind_system_inputs({"p": precond}, {"matrix": matrix})
    assert "matrix" in precond.bound
    assert "coordinates" not in precond.bound


def test_no_extras_declared_nothing_bound(matrix: torch.Tensor) -> None:
    """Preconditioners that declare no extra names are untouched."""
    precond = _TrackingPreconditioner(())
    _bind_system_inputs({"p": precond}, {"matrix": matrix})
    assert precond.bound == {}


def test_empty_preconditioners_dict_no_error() -> None:
    """An empty preconditioner map is handled without error."""
    _bind_system_inputs({}, {"matrix": torch.eye(3, dtype=torch.float64)})
