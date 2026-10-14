"""Zero-masks: support size, determinism and application."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.masking import apply_mask, random_support_mask

SEED = 123
DIM = 50


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.mark.parametrize(("density", "expected"), [(0.1, 5), (0.5, 25), (0.26, 13)])
def test_support_size_is_the_rounded_density(
    rng: np.random.Generator, density: float, expected: int
) -> None:
    mask = random_support_mask(DIM, density, rng)

    assert mask.dtype == np.bool_
    assert mask.shape == (DIM,)
    assert int(mask.sum()) == expected


def test_a_tiny_density_still_keeps_one_entry(rng: np.random.Generator) -> None:
    assert int(random_support_mask(DIM, 1e-6, rng).sum()) == 1


def test_the_support_has_no_repeated_entries(rng: np.random.Generator) -> None:
    """Drawing with replacement would leave fewer entries than the rounded density asks for."""
    for _ in range(20):
        assert int(random_support_mask(DIM, 0.5, rng).sum()) == DIM // 2


def test_the_same_generator_seed_gives_the_same_support() -> None:
    first = random_support_mask(DIM, 0.1, np.random.default_rng(SEED))
    second = random_support_mask(DIM, 0.1, np.random.default_rng(SEED))
    other = random_support_mask(DIM, 0.1, np.random.default_rng(SEED + 1))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_apply_mask_zeros_outside_the_support_and_keeps_values_inside() -> None:
    vector = np.arange(1.0, 6.0)
    mask = np.array([True, False, True, False, False])

    masked = apply_mask(vector, mask)

    np.testing.assert_array_equal(masked, [1.0, 0.0, 3.0, 0.0, 0.0])
    np.testing.assert_array_equal(vector, np.arange(1.0, 6.0))
