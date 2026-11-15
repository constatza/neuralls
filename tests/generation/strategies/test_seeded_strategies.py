"""Behaviour every seeded (rhs, solution) strategy shares, one table row per strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.interfaces import SolvedRows
from tests.generation_support import solved_rows


@dataclass(frozen=True, slots=True)
class SeededStrategy:
    """One strategy under test.

    Attributes:
        name: Registered strategy name.
        options: Extra config the strategy needs.
        rhs_is_a_times_solution: Whether ``b = A x`` is checked against the stored rows.
    """

    name: str
    options: dict[str, Any] = field(default_factory=dict)
    rhs_is_a_times_solution: bool = True


STRATEGIES = [
    SeededStrategy("krylov"),
    SeededStrategy("random"),
    SeededStrategy("eigenvector_forward"),
    SeededStrategy("eigenvector_inverse"),
    SeededStrategy("sinusoidal_field", {"omega": 1.0}, rhs_is_a_times_solution=False),
]


@pytest.fixture(params=STRATEGIES, ids=lambda strategy: strategy.name)
def strategy(request: pytest.FixtureRequest) -> SeededStrategy:
    """Each seeded strategy in turn."""
    return request.param


def _generate(
    strategy: SeededStrategy, matrix: np.ndarray, *, samples: int, seed: int
) -> SolvedRows:
    cfg = {"samples": samples, "seed": seed, **strategy.options}
    return solved_rows(run_generation(strategy.name, matrix, cfg=cfg))


@pytest.mark.parametrize("samples", [1, 4, 8])
def test_output_has_one_row_per_sample(
    strategy: SeededStrategy, spd_matrix: np.ndarray, samples: int
) -> None:
    result = _generate(strategy, spd_matrix, samples=samples, seed=0)

    n = spd_matrix.shape[0]
    assert result.rhs is not None
    assert result.solutions is not None
    assert result.rhs.shape == (samples, n)
    assert result.solutions.shape == (samples, n)


@pytest.mark.parametrize(
    "strategy",
    [strategy for strategy in STRATEGIES if strategy.rhs_is_a_times_solution],
    ids=lambda strategy: strategy.name,
)
def test_rhs_is_a_times_solution(strategy: SeededStrategy, spd_matrix: np.ndarray) -> None:
    result = _generate(strategy, spd_matrix, samples=5, seed=42)

    assert result.rhs is not None
    assert result.solutions is not None
    np.testing.assert_allclose(result.rhs, result.solutions @ spd_matrix.T, rtol=1e-12)


def test_same_seed_gives_identical_output(strategy: SeededStrategy, spd_matrix: np.ndarray) -> None:
    first = _generate(strategy, spd_matrix, samples=4, seed=7)
    second = _generate(strategy, spd_matrix, samples=4, seed=7)

    np.testing.assert_array_equal(first.rhs, second.rhs)
    np.testing.assert_array_equal(first.solutions, second.solutions)


def test_different_seeds_give_different_output(
    strategy: SeededStrategy, spd_matrix: np.ndarray
) -> None:
    first = _generate(strategy, spd_matrix, samples=4, seed=1)
    second = _generate(strategy, spd_matrix, samples=4, seed=2)

    assert not np.array_equal(first.rhs, second.rhs)
