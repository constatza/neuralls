"""Compile-time properties of known generation strategies.

Which strategies are file-backed archives, and what override key holds their glob.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from neuralls.shared.types import GenerationStrategyKind


@dataclass(frozen=True)
class _StrategyProperties:
    """Compile-time properties for a known generation strategy.

    Attributes:
        is_archive: Whether the strategy draws from a finite pool of files, each
            usable at most once. Archives never overload a matrix.
        glob_key: Override key holding the strategy's archive glob, or ``None`` when the
            strategy has no file-backed archive. The glob fixes the pool that explicit
            (matrix, file) indices refer to.
    """

    is_archive: bool
    glob_key: str | None = None


_STRATEGY_PROPERTIES: dict[GenerationStrategyKind, _StrategyProperties] = {
    GenerationStrategyKind.SOLUTION_ARCHIVE: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.RHS_ARCHIVE: _StrategyProperties(is_archive=True, glob_key="rhs_glob"),
    GenerationStrategyKind.SCALED_SOLUTIONS: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.VALIDATED_ARCHIVE: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.RESIDUALS: _StrategyProperties(
        is_archive=True, glob_key="solutions_glob"
    ),
    GenerationStrategyKind.GAUSSIAN_RESIDUALS: _StrategyProperties(is_archive=False),
}


def _properties_for(strategy_name: str) -> _StrategyProperties | None:
    """Properties of a known strategy, or ``None`` for a name outside the enum."""
    try:
        return _STRATEGY_PROPERTIES.get(GenerationStrategyKind(strategy_name))
    except ValueError:
        return None


def _archive_glob_for_strategy(
    strategy_name: str,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> str | None:
    """Return the strategy's configured archive glob, or ``None`` if it has none."""
    props = _properties_for(strategy_name)
    if props is None or props.glob_key is None:
        return None
    overrides = (strategy_overrides or {}).get(strategy_name, {})
    glob_pattern = overrides.get(props.glob_key)
    return glob_pattern if isinstance(glob_pattern, str) else None


def _is_solution_archive(strategy_name: str) -> bool:
    """Whether a strategy draws from a glob of solution vectors."""
    props = _properties_for(strategy_name)
    return props is not None and props.is_archive and props.glob_key == "solutions_glob"


def _explicit_solution_strategies(strategy_counts: Mapping[str, int]) -> frozenset[str]:
    """Solution-archive strategies with an explicit positive count.

    Their per-binding rows come from an explicit solution file by the cyclic map.
    """
    return frozenset(
        name for name, count in strategy_counts.items() if count > 0 and _is_solution_archive(name)
    )
