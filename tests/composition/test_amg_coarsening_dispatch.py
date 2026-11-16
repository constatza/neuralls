"""AMG coarsening dispatch accepts only known coarsening configs it can build faithfully."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from neuralls.composition.preconditioners.coarsening import _build_amg_coarsening
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import AMGPreconditionerConfig
from neuralls.shared.types import InnerProductKind

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


@pytest.fixture
def energy_pod_without_checkpoint() -> AMGPreconditionerConfig:
    """An energy POD naming an assignment but no checkpoint it could load."""
    return AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=Path("pod-fit"),
            rank=2,
            assignment="energy_fit",
            inner_product=InnerProductKind.ENERGY,
        )
    )


def test_an_energy_pod_without_a_checkpoint_is_not_fit_inline(
    identity_matrix: torch.Tensor, energy_pod_without_checkpoint: AMGPreconditionerConfig
) -> None:
    """The inline fit is Euclidean, so it must not stand in for an energy basis."""
    snapshots = np.eye(_MATRIX_SIZE)
    with pytest.raises(ValueError):
        _build_amg_coarsening(
            identity_matrix,
            energy_pod_without_checkpoint,
            staged_fit_data=(snapshots, snapshots),
        )
