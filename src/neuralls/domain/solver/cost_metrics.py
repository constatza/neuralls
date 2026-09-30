"""Size-normalized and throughput metrics derived from a ``CGComparisonResult``.

Raw ``generation_cost``/``setup_cost``/``solve_time_seconds``/``peak_memory_bytes``
are only comparable *within* one comparison run — they all share one system size ``n``.
Comparing them across comparisons run on different-sized matrices (the
data-generation pipeline produces many sizes) requires normalizing by ``n``
first. These are pure functions, not stored fields: ``n`` is a property of the
whole comparison (every preconditioner in one ``compare_preconditioners()``
call shares the same matrix), not of any individual result, so it is passed in
by the caller (``ComparisonResult.matrix_shape[0]``) rather than duplicated
onto every ``CGComparisonResult``.
"""

from __future__ import annotations

from neuralls.domain.solver.models.result import CGComparisonResult


def iterations_per_second(result: CGComparisonResult) -> float | None:
    """Iteration throughput, unnormalized by problem size.

    Args:
        result: Comparison result to derive throughput from.

    Returns:
        ``iterations / solve_time_seconds``, or ``None`` if solve time was
        never measured or no iterations ran.
    """
    if result.solve_time_seconds is None or result.iterations <= 0:
        return None
    return result.iterations / result.solve_time_seconds


def time_per_dof_per_iteration(result: CGComparisonResult, *, system_size: int) -> float | None:
    """Averaged per-iteration cost, normalized by problem size.

    The metric to compare across matrices of different sizes: a preconditioner
    that looks "fast" only because it ran on a smaller system collapses back
    to a comparable number here.

    Args:
        result: Comparison result to derive normalized cost from.
        system_size: Number of unknowns ``n`` in the linear system this
            result was solved on (``ComparisonResult.matrix_shape[0]``).

    Returns:
        ``avg_iteration_time_seconds / system_size``, or ``None`` if the
        average iteration time is unavailable.

    Raises:
        ValueError: If ``system_size`` is not positive.
    """
    if system_size <= 0:
        raise ValueError(f"system_size must be positive, got {system_size}.")
    avg_iteration_time = result.avg_iteration_time_seconds
    if avg_iteration_time is None:
        return None
    return avg_iteration_time / system_size


def setup_time_per_dof(result: CGComparisonResult, *, system_size: int) -> float | None:
    """Preconditioner construction cost, normalized by problem size.

    Args:
        result: Comparison result to derive normalized setup cost from.
        system_size: Number of unknowns ``n`` in the linear system.

    Returns:
        ``setup_cost.wall_time_seconds / system_size``, or ``None`` if setup
        cost was never measured (``result.setup_cost is None``).

    Raises:
        ValueError: If ``system_size`` is not positive.
    """
    if system_size <= 0:
        raise ValueError(f"system_size must be positive, got {system_size}.")
    setup_time_seconds = (
        result.setup_cost.wall_time_seconds if result.setup_cost is not None else None
    )
    if setup_time_seconds is None:
        return None
    return setup_time_seconds / system_size


def generation_time_per_dof(result: CGComparisonResult, *, system_size: int) -> float | None:
    """Dataset-generation cost feeding this preconditioner's fit/train step, normalized by problem size.

    Deliberately provenance-agnostic: returns a number whenever a
    ``StageCost`` is present on ``result.generation_cost``, regardless of
    whether its provenance is MEASURED/HISTORICAL/UNAVAILABLE.
    Provenance-based filtering (e.g. skipping UNAVAILABLE costs) is a
    presentation-layer concern (plots, MLflow logging), not this pure-math
    layer's job.

    Args:
        result: Comparison result to derive normalized generation cost from.
        system_size: Number of unknowns ``n`` in the linear system.

    Returns:
        ``generation_time_seconds / system_size``, or ``None`` if generation
        cost doesn't apply to this result.

    Raises:
        ValueError: If ``system_size`` is not positive.
    """
    if system_size <= 0:
        raise ValueError(f"system_size must be positive, got {system_size}.")
    generation_time_seconds = (
        result.generation_cost.wall_time_seconds if result.generation_cost is not None else None
    )
    if generation_time_seconds is None:
        return None
    return generation_time_seconds / system_size


def peak_memory_per_dof(result: CGComparisonResult, *, system_size: int) -> float | None:
    """Peak memory usage, normalized by problem size.

    Args:
        result: Comparison result to derive normalized memory from.
        system_size: Number of unknowns ``n`` in the linear system.

    Returns:
        ``peak_memory_bytes / system_size``, or ``None`` if memory was never
        measured.

    Raises:
        ValueError: If ``system_size`` is not positive.
    """
    if system_size <= 0:
        raise ValueError(f"system_size must be positive, got {system_size}.")
    if result.peak_memory_bytes is None:
        return None
    return result.peak_memory_bytes / system_size
