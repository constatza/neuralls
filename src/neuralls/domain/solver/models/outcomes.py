"""Per-system outcome of one run: solved with a recurring cost, or failed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Solved:
    """A run that reached the tolerance within the iteration cap.

    Attributes:
        iterations: Iterations taken.
        recurring_time_s: Generation + setup + solve wall time minus any
            one-time shared cost (a shared setup or generation is charged once,
            in ERT, the projection and break-even, not per system), or `None`
            when an unavailable cost makes it incomparable.
    """

    iterations: int
    recurring_time_s: float | None


@dataclass(frozen=True, slots=True)
class Failed:
    """A run stopped by the iteration cap or a breakdown: a right-censored observation.

    Carries no cost on purpose. What a failed run cost is the cap, not the
    price of a solution, and profiles keep it as unsolved rather than as a number.
    """


type SystemOutcome = Solved | Failed


type Outcomes = Mapping[str, Mapping[str, SystemOutcome]]
"""Preconditioner name -> {system id -> outcome}."""
