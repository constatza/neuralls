"""How one solved run is costed for profiles: iterations or recurring seconds."""

from __future__ import annotations

from collections.abc import Callable

from neuralls.domain.solver.models.outcomes import Solved

type Cost = Callable[[Solved], float | None]
"""The cost of a solved run in some unit; `None` when it is incomparable."""


def iterations(solved: Solved) -> float:
    """Cost in iterations: the iteration count of a solved run."""
    return float(solved.iterations)


def seconds(solved: Solved) -> float | None:
    """Recurring cost in seconds per system, `None` when incomparable.

    That is solve time plus any per-system setup or generation; a one-time
    shared cost (e.g. a checkpoint load) is excluded and charged once in ERT,
    the projection and break-even.

    A non-positive time carries no information and would break cost ratios and
    log axes, so it is incomparable too.
    """
    time = solved.recurring_time_s
    return time if time is not None and time > 0.0 else None
