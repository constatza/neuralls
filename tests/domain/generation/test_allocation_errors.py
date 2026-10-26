"""Invalid archive budgets fail for every archive strategy."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.generation.binding_allocation import _resolve_binding_strategy_counts
from neuralls.domain.generation.bindings import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec


@pytest.mark.parametrize(
    "strategy",
    ["solution_archive", "rhs_archive"],
)
def test_samples_all_without_glob_is_rejected(
    make_bindings: Callable[[int, int], list[SystemBinding]],
    strategy: str,
) -> None:
    """samples = -1 across several matrices requires a finite archive glob."""
    spec = DatasetSpec(mixture=MixtureSpec(counts={strategy: -1}, seed=0))

    with pytest.raises(ValueError):
        _resolve_binding_strategy_counts(
            bindings=make_bindings(2, 1),
            spec=spec,
            num_matrix_samples=2,
        )
