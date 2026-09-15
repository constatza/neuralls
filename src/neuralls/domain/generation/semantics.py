"""Semantic classification helpers for persisted generation outputs."""

from __future__ import annotations

from neuralls.shared.types import GenerationStrategyKind, RowKind

_NON_RESIDUAL_STRATEGIES: frozenset[GenerationStrategyKind] = frozenset(
    {
        GenerationStrategyKind.RANDOM,
        GenerationStrategyKind.NORMAL,
        GenerationStrategyKind.KRYLOV,
        GenerationStrategyKind.RHS_ARCHIVE,
        GenerationStrategyKind.SOLUTION_ARCHIVE,
        GenerationStrategyKind.VALIDATED_ARCHIVE,
        GenerationStrategyKind.SCALED_SOLUTIONS,
        GenerationStrategyKind.SPARSE_RHS,
        GenerationStrategyKind.EIGENVECTOR_FORWARD,
        GenerationStrategyKind.EIGENVECTOR_INVERSE,
        GenerationStrategyKind.GAUSSIAN_FORWARD,
        GenerationStrategyKind.GAUSSIAN_INVERSE,
        GenerationStrategyKind.UNIFORM_FORWARD,
        GenerationStrategyKind.UNIFORM_INVERSE,
        GenerationStrategyKind.CONSTANT_FORWARD,
        GenerationStrategyKind.CONSTANT_INVERSE,
        GenerationStrategyKind.NEUTRAL_ONES,
        GenerationStrategyKind.SMOOTHER_FILTERED_PROBES,
    }
)

_CG_INTERNAL_STRATEGIES: frozenset[GenerationStrategyKind] = frozenset(
    {
        GenerationStrategyKind.RESIDUALS,
        GenerationStrategyKind.GAUSSIAN_RESIDUALS,
        GenerationStrategyKind.SEARCH_DIRECTIONS,
    }
)


def classify_strategy_row_kind(kind: GenerationStrategyKind) -> RowKind:
    """Return the RowKind for all rows produced by this strategy (non-iter-0 rows).

    Args:
        kind: The generation strategy to classify.

    Returns:
        ``RowKind.STANDARD`` for externally-drawn (b, x) pairs;
        ``RowKind.CG_INTERNAL`` for CG-internal pairs (residuals, errors, or
        search-direction products — all unsafe for direct solver comparison).

    Raises:
        ValueError: If ``kind`` is not mapped to either category.
    """
    if kind in _NON_RESIDUAL_STRATEGIES:
        return RowKind.STANDARD
    if kind in _CG_INTERNAL_STRATEGIES:
        return RowKind.CG_INTERNAL
    raise ValueError(f"Unknown strategy kind: {kind!r}")
