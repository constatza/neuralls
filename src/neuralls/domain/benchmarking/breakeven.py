"""Break-even number of solved systems between a method and a reference.

Each method's projected cost of `n` solved systems is the line
`T(n) = a + b * n` (`ProjectedComparisonResult.projected_total_seconds`) with
`a` the one-time cost and `b` the recurring cost per solved system. Two such
lines cross at most once, so the comparison has exactly four outcomes, the
`BreakEven` variants. Costs are measured quantities of the sweep, extrapolated
in closed form; nothing is run at `n` systems.
"""

from __future__ import annotations

from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.statistics import (
    AlwaysCheaper,
    BreakEven,
    BreakEvenAt,
    CheaperOnlyBelow,
    NeverCheaper,
)


def break_even(
    method: ProjectedComparisonResult, reference: ProjectedComparisonResult
) -> BreakEven:
    """Where the method's total cost crosses the reference's, in solved systems.

    A method with a lower recurring cost wins from the crossing point on
    (`BreakEvenAt`), or from the start when its one-time cost is not higher
    (`AlwaysCheaper`). A method with a higher recurring cost can only win
    below the crossing point (`CheaperOnlyBelow`, needs a lower one-time
    cost) and otherwise never does (`NeverCheaper`). Equal recurring costs
    decide on the one-time cost alone: a lower one is `AlwaysCheaper`, an
    equal or higher one `NeverCheaper`.

    Args:
        method: The method under test.
        reference: The method it is compared against.

    Returns:
        The break-even outcome of `method` against `reference`.
    """
    one_time_method, one_time_reference = method.one_time_seconds, reference.one_time_seconds
    recurring_method = method.per_solved_system_seconds
    recurring_reference = reference.per_solved_system_seconds
    if recurring_method < recurring_reference:
        if one_time_method <= one_time_reference:
            return AlwaysCheaper()
        return BreakEvenAt(
            (one_time_method - one_time_reference) / (recurring_reference - recurring_method)
        )
    if one_time_method >= one_time_reference:
        return NeverCheaper()
    if recurring_method == recurring_reference:
        return AlwaysCheaper()
    return CheaperOnlyBelow(
        (one_time_reference - one_time_method) / (recurring_method - recurring_reference)
    )
