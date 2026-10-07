"""Strategy sample counts, and trace-row counts for trajectory-harvesting strategies."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from .step_window import StepWindow


def rounded_counts(total: int, proportions: Mapping[str, float]) -> dict[str, int]:
    """Convert strategy proportions to integer counts that sum exactly to total.

    The proportions do not need to sum to 1.0; they are normalized internally.

    Args:
        total: Total number of samples to generate
        proportions: Mapping of strategy names to proportions (non-negative)

    Returns:
        Dictionary mapping strategy names to integer counts

    Raises:
        ValueError: If total <= 0, proportions empty, or weights negative/zero sum
    """
    if total <= 0:
        raise ValueError(f"Total samples must be positive, got {total}")
    if not proportions:
        raise ValueError("At least one strategy proportion must be provided")

    weights: dict[str, float] = {}
    for key, value in proportions.items():
        weight = float(value)
        if weight < 0:
            raise ValueError(f"Mix weight for '{key}' must be non-negative, got {value}")
        weights[key] = weight

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Sum of mix weights must be positive")

    scaled = {key: (weight / total_weight) * total for key, weight in weights.items()}
    counts = {key: math.floor(amount) for key, amount in scaled.items()}
    remainders = {key: scaled[key] - counts[key] for key in scaled}

    remaining = total - sum(counts.values())
    if remaining > 0:
        sorted_keys = sorted(
            remainders.keys(),
            key=lambda key: (-remainders[key], -weights[key], key),
        )
        idx = 0
        while remaining > 0 and sorted_keys:
            key = sorted_keys[idx % len(sorted_keys)]
            counts[key] += 1
            remaining -= 1
            idx += 1

    return counts


def resolve_strategy_counts(
    counts: Mapping[str, int] | None,
    mix: Mapping[str, float] | None,
    total: int | None,
) -> dict[str, int]:
    """Resolve strategy counts from explicit counts or mix/total pair.

    Pure function: no side effects.

    Args:
        counts: Optional explicit strategy counts
        mix: Optional strategy proportions
        total: Total samples (required if mix provided)

    Returns:
        Dictionary of strategy_name -> count (only positive counts)

    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    if counts is not None and mix is not None:
        raise ValueError("Specify either explicit counts or a mix/total pair, not both")

    if counts is None:
        if mix is None:
            raise ValueError("Either counts or mix must be provided")
        if total is None:
            raise ValueError("Parameter 'total' is required when using mix")
        resolved = rounded_counts(int(total), mix)
    else:
        resolved = {name: int(value) for name, value in counts.items()}

    # Filter out zero counts (but keep -1 which means "all available")
    nonzero = {name: value for name, value in resolved.items() if value != 0}
    if not nonzero:
        raise ValueError("No strategy counts were provided")

    return nonzero


def _build_trace_indices(
    sample_idx: int,
    indices: range,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sample and iteration index arrays for trace data.

    Pure function. `indices` is an already-resolved set of trajectory step
    indices (e.g. from `StepWindow.select_with_indices`), not a `StepWindow`
    — this keeps the "which rows were selected" and "what are their
    original indices" pairing structurally impossible to compute from two
    different arrays by mistake (see `step_window.py`'s
    `select_with_indices` docstring).

    Args:
        sample_idx: Sample index for this trace.
        indices: The trajectory step indices that were kept.

    Returns:
        Tuple of (sample_indices, iteration_indices).
    """
    iteration_indices = np.fromiter(indices, dtype=np.int64)
    return (
        np.full(iteration_indices.shape[0], sample_idx, dtype=np.int64),
        iteration_indices,
    )


def trace_rows_per_system(window: StepWindow) -> int:
    """Return the worst-case number of kept trace rows for one base system.

    Computed against `window.stop + 1` — the trajectory length when the
    safety cap is fully used, which is always the case for trajectories:
    the solver runs exactly `window.stop` steps, so every base system yields
    exactly this many rows. Used only to budget how many base systems to run up front — see
    `resolve_trace_generation_counts`.
    """
    return len(window.resolve_indices(window.stop + 1))


class _WindowedConfig(Protocol):
    """A validated strategy config that exposes its trajectory window."""

    @property
    def window(self) -> StepWindow: ...


def trace_rows_per_base_system(
    config_type: Callable[..., _WindowedConfig],
) -> Callable[[Mapping[str, Any]], int]:
    """Build the K_rows callable for a trace strategy from its config model."""

    def _rows(cfg: Mapping[str, Any]) -> int:
        return trace_rows_per_system(config_type(**cfg).window)

    return _rows


def required_trace_systems(samples: int, *, window: StepWindow) -> int:
    """Return the base-system count for a desired trace-row budget."""
    rows_per_system = trace_rows_per_system(window)
    return max(1, math.ceil(samples / rows_per_system))


def resolve_trace_generation_counts(
    samples: int,
    *,
    window: StepWindow,
    available_systems: int | None,
    strategy_name: str,
) -> tuple[int, int | None]:
    """Resolve base-system count for trajectory-harvesting strategies."""
    if samples == -1:
        if available_systems is None:
            raise ValueError(
                f"Strategy '{strategy_name}' does not support samples=-1 without "
                "a finite archive-backed source."
            )
        return available_systems, None
    return required_trace_systems(samples, window=window), samples
