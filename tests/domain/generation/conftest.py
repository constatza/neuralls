"""Shared fixtures for generation-domain allocation tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import MixtureSpec


@pytest.fixture
def make_bindings() -> Callable[[int, int], list[SystemBinding]]:
    """Build bindings for ``num_matrices`` matrices with ``per_matrix`` bindings each.

    Matrix ``m`` owns the consecutive binding positions it is given, so the
    returned list is ordered by matrix.
    """

    def _build(num_matrices: int, per_matrix: int) -> list[SystemBinding]:
        return [
            SystemBinding(sample_id=m * per_matrix + k, matrix_sample_id=m)
            for m in range(num_matrices)
            for k in range(per_matrix)
        ]

    return _build


@pytest.fixture
def make_mixture() -> Callable[..., MixtureSpec]:
    """Build a single-strategy ``MixtureSpec``, optionally overriding the solutions glob."""

    def _build(strategy: str, count: int, *, glob: str | None = None) -> MixtureSpec:
        overrides = {strategy: {"solutions_glob": glob}} if glob is not None else None
        return MixtureSpec(counts={strategy: count}, seed=0, strategy_overrides=overrides)

    return _build
