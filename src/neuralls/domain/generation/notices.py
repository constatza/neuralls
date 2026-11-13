"""Generation outcomes an operator must see, returned as data instead of logged.

The domain never logs. A strategy, the allocation or the scale aggregation that finds
something worth reporting returns it as a notice beside its result; composition renders
every notice with an exhaustive ``match`` and logs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RowCapSource(StrEnum):
    """What limited a strategy's rows below its request."""

    ARCHIVE_PAIRS = "archive (matrix, file) pairs"
    """The request is a run total; the archive supplies M*K pairs times rows per pair."""
    SOLUTION_FILE = "solution file"
    """The request is per binding; the explicit solution source supplies K rows."""


@dataclass(frozen=True, slots=True)
class RowsCapped:
    """A strategy asked for more rows than its source holds and was capped.

    Attributes:
        strategy: Strategy name as the mixture spelled it.
        requested_rows: Rows the strategy asked for, in the unit its source implies
            (a run total for archive pairs, per binding for a solution file).
        emitted_rows: Rows it will emit after the cap, in the same unit.
        source: What the cap came from.
    """

    strategy: str
    requested_rows: int
    emitted_rows: int
    source: RowCapSource


@dataclass(frozen=True, slots=True)
class SolveNotConverged:
    """One row's iterative solve stopped before reaching its tolerance.

    Attributes:
        row: Row position within the strategy's output.
        exit_code: Solver exit code (positive: iterations exhausted; negative: breakdown).
        residual: Absolute residual norm of the returned solution.
    """

    row: int
    exit_code: int
    residual: float


@dataclass(frozen=True, slots=True)
class InvalidPairs:
    """Archive (solution, RHS) pairs that fail the consistency check but were kept.

    Attributes:
        count: Pairs above the tolerance.
        total: Pairs checked.
        max_error: Largest relative residual found.
        tolerance: Relative residual the pairs were checked against.
    """

    count: int
    total: int
    max_error: float
    tolerance: float


type SolveNotice = SolveNotConverged | InvalidPairs
"""Outcomes found while a strategy produced its rows; they travel with those rows."""


__all__ = [
    "InvalidPairs",
    "RowCapSource",
    "RowsCapped",
    "SolveNotConverged",
    "SolveNotice",
]
