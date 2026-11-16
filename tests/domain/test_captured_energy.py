"""The snapshot energy a POD basis captures, read from the basis alone."""

from __future__ import annotations

import pytest
import torch
from torchalg.utils.pod_basis import compute_pod_basis

from neuralls.domain.linalg import captured_energy_fraction

RANK = 2


@pytest.fixture
def snapshots() -> torch.Tensor:
    """Six snapshots of a five-dof problem with spread-out singular values."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(6, 5, generator=generator, dtype=torch.float64)


@pytest.fixture
def weight() -> torch.Tensor:
    """A dense SPD inner-product weight."""
    generator = torch.Generator().manual_seed(1)
    factor = torch.randn(5, 5, generator=generator, dtype=torch.float64)
    return factor @ factor.T + 5 * torch.eye(5, dtype=torch.float64)


@pytest.fixture
def row_scales() -> torch.Tensor:
    """Per-snapshot scales."""
    return torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.25], dtype=torch.float64)


def _kept_share(singular_values: torch.Tensor) -> float:
    energy = singular_values.square()
    return float(energy[:RANK].sum() / energy.sum())


def test_euclidean_fraction_is_the_kept_share_of_squared_singular_values(
    snapshots: torch.Tensor,
) -> None:
    basis = compute_pod_basis(snapshots, RANK)

    expected = _kept_share(torch.linalg.svdvals(snapshots))
    assert captured_energy_fraction(snapshots, basis) == pytest.approx(expected)


def test_row_scaled_fraction_is_that_of_the_scaled_snapshots(
    snapshots: torch.Tensor, row_scales: torch.Tensor
) -> None:
    basis = compute_pod_basis(snapshots, RANK, row_scales=row_scales)

    expected = _kept_share(torch.linalg.svdvals(snapshots * row_scales.unsqueeze(-1)))
    assert captured_energy_fraction(snapshots, basis, row_scales) == pytest.approx(expected)


def test_weighted_fraction_is_measured_in_the_weight_norm(
    snapshots: torch.Tensor, weight: torch.Tensor
) -> None:
    basis = compute_pod_basis(snapshots, RANK, inner_product=weight)

    expected = _kept_share(torch.linalg.svdvals(snapshots @ torch.linalg.cholesky(weight)))
    assert captured_energy_fraction(snapshots, basis, weight=weight) == pytest.approx(expected)


def test_a_full_basis_captures_all_the_energy(snapshots: torch.Tensor) -> None:
    basis = compute_pod_basis(snapshots, 5)

    assert captured_energy_fraction(snapshots, basis) == pytest.approx(1.0)
