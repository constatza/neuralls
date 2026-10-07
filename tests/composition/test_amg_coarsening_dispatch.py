"""AMG coarsening dispatch accepts only known coarsening configs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from neuralls.composition.preconditioners.coarsening import _build_amg_coarsening
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
