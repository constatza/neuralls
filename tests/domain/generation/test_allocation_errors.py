"""Error messages from budget resolution name the glob key the strategy actually uses."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.generation.binding_allocation import _resolve_binding_strategy_counts
from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec


@pytest.mark.parametrize(
    ("strategy", "glob_key"),
    [
        ("solution_archive", "solutions_glob"),
        ("rhs_archive", "rhs_glob"),
    ],
)
def test_samples_all_without_glob_names_the_strategy_glob_key(
    make_bindings: Callable[[int, int], list[SystemBinding]],
    strategy: str,
    glob_key: str,
) -> None:
    """samples = -1 across several matrices with no glob fails and names that strategy's key."""
    spec = DatasetSpec(mixture=MixtureSpec(counts={strategy: -1}, seed=0))

    with pytest.raises(ValueError) as excinfo:
        _resolve_binding_strategy_counts(
            bindings=make_bindings(2, 1),
            spec=spec,
            num_matrix_samples=2,
        )

    message = str(excinfo.value)
    assert f"'{glob_key}'" in message
    other_key = "rhs_glob" if glob_key == "solutions_glob" else "solutions_glob"
    assert other_key not in message
