"""Row-budget plan for one generation run, fixed before any sample is generated.

The plan is the resolved per-binding, per-strategy row counts. It carries no
sample data, so the writer can size its outputs from it before generation starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

ALL_SAMPLES: int = -1
"""Sentinel strategy count: emit all available samples for this binding.

Its row total is only known after generation, so a plan holding it is not exact.
"""


@dataclass(frozen=True)
class BindingAllocation:
    """Per-binding strategy budgets resolved from one global count map.

    Attributes:
        counts: Per-binding strategy count maps, parallel to the bindings.
        file_indices: Per-binding explicit archive-file indices, parallel to the bindings.
            Only archive strategies with a glob carry entries. Indices are positions in
            the archive pool after the strategy's configured skip. Bindings of different
            matrices never share a (matrix, file) pair.
    """

    counts: tuple[dict[str, int], ...]
    file_indices: tuple[dict[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class StrategyRows:
    """Planned row count of one strategy on one binding.

    Attributes:
        name: Strategy name.
        rows: Planned row count, or ``ALL_SAMPLES`` when the count is open-ended.
    """

    name: str
    rows: int


@dataclass(frozen=True)
class BindingPlan:
    """Planned strategy rows for one binding, in strategy order.

    Attributes:
        binding_index: Position of the binding in the run's binding sequence.
        strategies: Planned rows per active strategy, in the order they are generated.
    """

    binding_index: int
    strategies: tuple[StrategyRows, ...]

    @property
    def total_rows(self) -> int:
        """Planned rows for this binding, summed over its strategies.

        Raises:
            ValueError: If a strategy count is open-ended.
        """
        if any(strategy.rows == ALL_SAMPLES for strategy in self.strategies):
            raise ValueError("Binding has an open-ended strategy count; its row total is unknown.")
        return sum(strategy.rows for strategy in self.strategies)


@dataclass(frozen=True)
class BatchPlan:
    """Immutable row budget of one generation run.

    Attributes:
        bindings: One plan per binding that contributes rows, in binding order.
    """

    bindings: tuple[BindingPlan, ...]

    @property
    def is_exact(self) -> bool:
        """Whether every planned count is a concrete row count."""
        return all(
            strategy.rows != ALL_SAMPLES
            for binding in self.bindings
            for strategy in binding.strategies
        )

    def require_exact(self) -> None:
        """Raise when any strategy count is open-ended, naming the strategies.

        Streamed writers size their file grid from the plan, so an open-ended count
        has no grid and must be fixed in the configuration.

        Raises:
            ValueError: If any strategy count is ``ALL_SAMPLES``.
        """
        open_ended = sorted(
            {
                strategy.name
                for binding in self.bindings
                for strategy in binding.strategies
                if strategy.rows == ALL_SAMPLES
            }
        )
        if open_ended:
            raise ValueError(
                "Streamed generation needs an exact row count for every strategy; "
                f"open-ended strategies: {', '.join(open_ended)}. Set a concrete count."
            )

    @property
    def total_rows(self) -> int:
        """Total planned rows across every binding and strategy.

        Raises:
            ValueError: If the plan holds an open-ended ``ALL_SAMPLES`` count.
        """
        if not self.is_exact:
            raise ValueError("Plan has an open-ended strategy count; its row total is unknown.")
        return sum(strategy.rows for binding in self.bindings for strategy in binding.strategies)

    @property
    def strategy_totals(self) -> Mapping[str, int]:
        """Planned rows per strategy, summed over bindings.

        Raises:
            ValueError: If the plan holds an open-ended ``ALL_SAMPLES`` count.
        """
        if not self.is_exact:
            raise ValueError("Plan has an open-ended strategy count; its row totals are unknown.")
        totals: dict[str, int] = {}
        for binding in self.bindings:
            for strategy in binding.strategies:
                totals[strategy.name] = totals.get(strategy.name, 0) + strategy.rows
        return totals


def plan_batches(allocation: BindingAllocation) -> BatchPlan:
    """Build the immutable row plan from resolved per-binding counts.

    Pure: the allocation already fixes every count, so the plan only reshapes it.
    Bindings whose counts are all zero emit nothing, so they are omitted and keep their
    position index. A planned binding is therefore one that contributes rows to the run.

    Args:
        allocation: Per-binding strategy counts resolved by the orchestration layer.

    Returns:
        BatchPlan listing the planned rows for every emitting binding.
    """
    binding_plans = tuple(
        BindingPlan(
            binding_index=binding_index,
            strategies=tuple(StrategyRows(name=name, rows=rows) for name, rows in counts.items()),
        )
        for binding_index, counts in enumerate(allocation.counts)
        if any(rows != 0 for rows in counts.values())
    )
    return BatchPlan(bindings=binding_plans)


__all__ = [
    "ALL_SAMPLES",
    "BatchPlan",
    "BindingAllocation",
    "BindingPlan",
    "StrategyRows",
    "plan_batches",
]
