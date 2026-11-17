"""`PreconditionerService` builds the implementation matching the matrix's layout."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from dlkit.infrastructure.types.split import IndexSplit
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
from tests.fakes.split_store import InMemorySplitStore

FIT_DIR = Path("pod-fit")


@pytest.fixture
def diagonal_matrix() -> torch.Tensor:
    """Well-conditioned 4x4 SPD diagonal matrix."""
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float64))


@pytest.fixture
def service() -> PreconditionerService:
    return PreconditionerService(FakeFitData(), InMemorySplitStore())


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


@pytest.fixture
def five_row_fit_data() -> FakeFitData:
    """Rows whose values are their dataset row, so a staged row names where it came from."""
    rows = np.arange(5.0)[:, None]
    return FakeFitData(rhs=rows, solutions=10 * rows)


@pytest.fixture
def pinned_split() -> InMemorySplitStore:
    return InMemorySplitStore(
        {FIT_DIR: IndexSplit(train=(0, 3), validation=(1,), test=(2, 4), predict=None)}
    )


def test_an_inline_pod_fit_stages_only_the_train_fold(
    monkeypatch: pytest.MonkeyPatch,
    diagonal_matrix: torch.Tensor,
    five_row_fit_data: FakeFitData,
    pinned_split: InMemorySplitStore,
) -> None:
    """Comparisons draw held-out systems from the test fold, so the fit never sees them."""
    staged: list[tuple[np.ndarray, ...] | None] = []

    def capture(*_args: object, staged_fit_data: tuple[np.ndarray, ...] | None, **_kw: object):
        staged.append(staged_fit_data)

    monkeypatch.setattr(_preconditioner_setup, "create_preconditioner", capture)
    config = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR))
    service = PreconditionerService(five_row_fit_data, pinned_split)

    service.preload(diagonal_matrix, config)
    service.create_preconditioner(diagonal_matrix, config)

    assert [[array.ravel().tolist() for array in arrays or ()] for arrays in staged] == [
        [[0.0, 3.0], [0.0, 30.0]]
    ]
