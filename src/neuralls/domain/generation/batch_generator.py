"""Stream a generation run as bounded batches, in binding order then strategy order."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .batch import SampleBatch
from .batch_plan import BatchPlan

StrategyRunner = Callable[[int, str], SampleBatch]
"""Generate one strategy's whole output for one binding.

Called as ``runner(binding_index, strategy_name)``. The returned batch may hold zero rows,
for example when a trace strategy has no base systems left to emit.
"""


def generate_batches(
    plan: BatchPlan,
    run_strategy: StrategyRunner,
    *,
    batch_size: int,
) -> Iterator[SampleBatch]:
    """Yield every planned strategy output as chunks of at most ``batch_size`` rows.

    Order is binding order, then the plan's strategy order, then row order within the
    strategy output. Empty outputs yield nothing.

    Args:
        plan: Row budget listing the bindings and strategies to run.
        run_strategy: Produces one strategy's output for one binding.
        batch_size: Maximum rows per yielded batch. Must be positive.

    Yields:
        SampleBatch chunks carrying their binding index and strategy name.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for binding in plan.bindings:
        for strategy in binding.strategies:
            output = run_strategy(binding.binding_index, strategy.name)
            for start in range(0, len(output), batch_size):
                yield output.slice(start, start + batch_size)


__all__ = ["StrategyRunner", "generate_batches"]
