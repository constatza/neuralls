"""AMG coarsening dispatch accepts only known coarsening configs it can build faithfully."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from neuralls.composition.preconditioners.coarsening import (
    MissingPODBasisError,
    _build_amg_coarsening,
)
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import AMGPreconditionerConfig

_MATRIX_SIZE = 4


@pytest.fixture
def identity_matrix() -> torch.Tensor:
    """Small dense SPD matrix; the dispatch rejects the config before using it."""
    return torch.eye(_MATRIX_SIZE, dtype=torch.float64)


def test_unknown_coarsening_config_is_rejected(identity_matrix: torch.Tensor) -> None:
    """A coarsening that is not a known config type must fail, not fall through as aggregation."""
    config = cast(AMGPreconditionerConfig, SimpleNamespace(coarsening=object()))
    with pytest.raises(TypeError):
        _build_amg_coarsening(identity_matrix, config)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
def test_a_pod_build_without_its_fitted_basis_is_rejected(
    identity_matrix: torch.Tensor, sparse: bool
) -> None:
    """The basis is fitted once per method before the timed setups, never inside one."""
    config = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=Path("pod-fit"), rank=2)
    )
    matrix = identity_matrix.to_sparse_csr() if sparse else identity_matrix

    with pytest.raises(MissingPODBasisError) as caught:
        _build_amg_coarsening(matrix, config, sparse=sparse)

    assert caught.value.dataset_dir == Path("pod-fit")
