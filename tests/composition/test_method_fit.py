"""A POD method's basis is fitted once, on its train fold, and the fit is measured."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from dlkit.infrastructure.types.split import IndexSplit
from scipy.sparse import csr_array
from torchalg.utils.pod_basis import compute_pod_basis

from neuralls.composition.comparison._method_fit import (
    FittedPOD,
    MethodFitFailed,
    NoMethodFit,
    fit_method,
)
from neuralls.domain.solver.models.cost import Known
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.types import CostProvenance, InnerProductKind
from tests.fakes.fit_data import FakeFitData
from tests.fakes.split_store import InMemorySplitStore
from tests.loaders_support import CPU

FIT_DIR = Path("pod-fit")
N_DOFS = 4
DIAGONALS = {0: (4.0, 3.0, 2.0, 1.0), 1: (2.0, 5.0, 1.0, 3.0), 2: (90.0, 90.0, 90.0, 90.0)}
"""Matrices 0 and 1 are bound to train rows; matrix 2 only to the held-out test row."""


@pytest.fixture
def fit_data() -> FakeFitData:
    """Seven seeded snapshots; the last two rows are held out and bind their own matrix."""
    solutions = np.random.default_rng(3).standard_normal((7, N_DOFS))
    return FakeFitData(
        rhs=np.zeros_like(solutions),
        solutions=solutions,
        matrix_index=np.array([0, 0, 1, 1, 0, 2, 2]),
        matrices={key: csr_array(np.diag(diagonal)) for key, diagonal in DIAGONALS.items()},
    )


@pytest.fixture
def splits() -> InMemorySplitStore:
    return InMemorySplitStore(
        {FIT_DIR: IndexSplit(train=(0, 1, 2, 3, 4), validation=(), test=(5, 6), predict=None)}
    )


def _pod(rank: int, inner_product: InnerProductKind) -> AMGPreconditionerConfig:
    return AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, rank=rank, inner_product=inner_product)
    )


@pytest.mark.parametrize(
    ("inner_product", "weight"),
    [
        (InnerProductKind.EUCLIDEAN, np.eye(N_DOFS)),
        (InnerProductKind.ENERGY, np.diag((3.0, 4.0, 1.5, 2.0))),
    ],
    ids=["euclidean", "energy-weighted-by-the-train-mean"],
)
def test_a_pod_fit_is_orthonormal_in_its_inner_product_and_measured(
    fit_data: FakeFitData,
    splits: InMemorySplitStore,
    inner_product: InnerProductKind,
    weight: np.ndarray,
) -> None:
    fit = fit_method(_pod(2, inner_product), fit_data=fit_data, splits=splits, device=CPU)

    assert isinstance(fit, FittedPOD)
    gram = fit.basis.mT @ torch.as_tensor(weight) @ fit.basis
    torch.testing.assert_close(gram, torch.eye(2, dtype=torch.float64))
    assert isinstance(fit.cost, Known)
    assert fit.cost.provenance is CostProvenance.MEASURED


@pytest.mark.parametrize(("n_snapshots", "rows"), [(-1, 5), (3, 3)], ids=["all", "cut"])
def test_a_pod_fit_reads_only_the_train_fold_cut_to_its_snapshot_count(
    fit_data: FakeFitData, splits: InMemorySplitStore, n_snapshots: int, rows: int
) -> None:
    """Comparisons draw held-out systems from the test fold, so the fit never sees them."""
    pod = PODCoarseningConfig(dataset_dir=FIT_DIR, rank=2, n_snapshots=n_snapshots)

    fit = fit_method(
        AMGPreconditionerConfig(coarsening=pod), fit_data=fit_data, splits=splits, device=CPU
    )

    assert isinstance(fit, FittedPOD)
    expected = compute_pod_basis(torch.as_tensor(fit_data.solutions[:rows]), 2)
    torch.testing.assert_close(fit.basis, expected, rtol=0.0, atol=0.0)


def test_a_rank_above_the_snapshot_dimensions_is_a_measured_fit_failure(
    fit_data: FakeFitData, splits: InMemorySplitStore
) -> None:
    fit = fit_method(
        _pod(N_DOFS + 1, InnerProductKind.EUCLIDEAN), fit_data=fit_data, splits=splits, device=CPU
    )

    assert isinstance(fit, MethodFitFailed)
    assert isinstance(fit.error, ValueError)
    assert fit.cost.provenance is CostProvenance.MEASURED


@pytest.mark.parametrize(("rank", "warnings"), [(1, 1), (N_DOFS, 0)], ids=["low", "full"])
def test_a_rank_capturing_little_snapshot_energy_is_warned_about(
    fit_data: FakeFitData,
    splits: InMemorySplitStore,
    captured_log: Callable[[str], list[str]],
    rank: int,
    warnings: int,
) -> None:
    messages = captured_log("WARNING")

    fit_method(_pod(rank, InnerProductKind.EUCLIDEAN), fit_data=fit_data, splits=splits, device=CPU)

    assert len(messages) == warnings


def test_a_method_without_a_pod_basis_has_nothing_to_fit(
    fit_data: FakeFitData, splits: InMemorySplitStore
) -> None:
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)

    assert fit_method(jacobi, fit_data=fit_data, splits=splits, device=CPU) == NoMethodFit()
    assert fit_data.dense_reads == []
