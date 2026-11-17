"""`PreconditionerService` builds the implementation matching the matrix's layout."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torchalg.preconditioners.implementations import JacobiPreconditioner
from torchalg.sparse.preconditioners.jacobi import (
    JacobiPreconditioner as SparseJacobiPreconditioner,
)

from neuralls.composition.comparison import _preconditioner_setup
from neuralls.composition.comparison._preconditioner_setup import PreconditionerService
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from tests.fakes.fit_data import FakeFitData

FIT_DIR = Path("pod-fit")


@pytest.fixture
def diagonal_matrix() -> torch.Tensor:
    """Well-conditioned 4x4 SPD diagonal matrix."""
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float64))


@pytest.fixture
def service() -> PreconditionerService:
    return PreconditionerService(FakeFitData())


@pytest.mark.parametrize(
    ("to_layout", "expected"),
    [
        (torch.Tensor.to_sparse_csr, SparseJacobiPreconditioner),
        (torch.Tensor.contiguous, JacobiPreconditioner),
    ],
    ids=["csr", "dense"],
)
def test_service_builds_the_implementation_of_the_matrix_layout(
    to_layout: Callable[[torch.Tensor], torch.Tensor],
    expected: type,
    diagonal_matrix: torch.Tensor,
    service: PreconditionerService,
) -> None:
    config = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)

    precond = service.create_preconditioner(to_layout(diagonal_matrix), config)

    assert type(precond) is expected


def test_every_build_of_a_pod_method_borrows_its_fitted_basis(
    monkeypatch: pytest.MonkeyPatch, diagonal_matrix: torch.Tensor
) -> None:
    """The basis is fitted once per method; a build only forms its transfer from it."""
    handed: list[torch.Tensor | None] = []

    def capture(*_args: object, pod_basis: torch.Tensor | None, **_kw: object) -> None:
        handed.append(pod_basis)

    monkeypatch.setattr(_preconditioner_setup, "create_preconditioner", capture)
    config = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR))
    basis = torch.eye(4, 2, dtype=torch.float64)
    service = PreconditionerService(FakeFitData(), pod_basis=basis)

    for _ in range(2):
        service.preload(diagonal_matrix, config)
        service.create_preconditioner(diagonal_matrix, config)

    assert [given is basis for given in handed] == [True, True]
