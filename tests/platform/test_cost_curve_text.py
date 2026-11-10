"""Legend and axis wording of the break-even and convergence-factor figures, as pure functions."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.cost_curves import BreakEvenState
from neuralls.platform.reporting.seaborn_cost_curves import (
    InconsistentBreakEvenError,
    break_even_legend,
    reference_curve_legend,
)


@pytest.mark.parametrize(
    ("state", "break_even_rhs", "expected"),
    [
        (BreakEvenState.BREAKS_EVEN, 18.75, "AMG: breaks even at n* = 18.8 RHS"),
        (BreakEvenState.BREAKS_EVEN, 0.0, "AMG: breaks even at n* = 0.0 RHS"),
        (BreakEvenState.NEVER, None, "AMG: never breaks even (not faster per RHS)"),
        (
            BreakEvenState.NO_COMMON_SYSTEMS,
            None,
            "AMG: no systems in common with the reference",
        ),
    ],
)
def test_a_compared_method_states_where_or_whether_it_breaks_even(
    state: BreakEvenState, break_even_rhs: float | None, expected: str
) -> None:
    assert break_even_legend("AMG", state, break_even_rhs) == expected


@pytest.mark.parametrize(
    ("state", "break_even_rhs"),
    [
        (BreakEvenState.BREAKS_EVEN, None),
        (BreakEvenState.NEVER, 3.0),
        (BreakEvenState.NO_COMMON_SYSTEMS, 3.0),
    ],
)
def test_a_break_even_count_contradicting_its_state_is_rejected(
    state: BreakEvenState, break_even_rhs: float | None
) -> None:
    with pytest.raises(InconsistentBreakEvenError):
        break_even_legend("AMG", state, break_even_rhs)


def test_the_reference_curve_names_the_method_whose_systems_its_means_share() -> None:
    assert (
        reference_curve_legend("Jacobi", "AMG")
        == "Jacobi (reference, on the systems shared with AMG)"
    )
