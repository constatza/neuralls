"""Count-based generation strategy plan parsing.

This module replaces percentage-based mixing with explicit sample counts.
Each strategy specifies either:
  - samples > 0: Exact count
  - samples = -1: All available (for archives)
  - samples = 0: Skip this strategy
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neuralls.shared.types import GenerationStrategyKind

_RHS_ARCHIVE = "rhs_archive"
_SOLUTION_ARCHIVE = "solution_archive"


def canonicalize_strategy_name(raw_name: str) -> str:
    """Normalize strategy identifiers while preserving legacy aliases.

    Args:
        raw_name: Strategy name from config (e.g., "rhs-archive", "RHS_ARCHIVE")

    Returns:
        Canonical name (e.g., "rhs_archive")

    Examples:
        >>> canonicalize_strategy_name("rhs-archive")
        'rhs_archive'
        >>> canonicalize_strategy_name("solution_bank")
        'solution_archive'
    """
    normalized = raw_name.strip().lower().replace("-", "_")

    if normalized in {"rhs_bank", "rhs_repository"}:
        return _RHS_ARCHIVE
    if normalized in {"solution_bank", "solution_repository"}:
        return _SOLUTION_ARCHIVE

    return normalized


def canonicalize_strategy_kind(raw_name: str) -> GenerationStrategyKind:
    """Normalize a raw strategy identifier into the internal enum."""
    normalized = canonicalize_strategy_name(raw_name)
    try:
        return GenerationStrategyKind(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported generation strategy '{raw_name}'.") from exc


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Immutable specification for a single generation strategy.

    Attributes:
        raw_name: Original name from config
        canonical_name: Normalized name for registry lookup
        samples: Sample count (>0 exact, -1 all, 0 skip)
        options: Additional strategy-specific parameters
    """

    raw_name: str
    kind: GenerationStrategyKind
    samples: int
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate count constraints."""
        if self.samples < -1:
            raise ValueError(
                f"Strategy '{self.raw_name}' has invalid samples count: {self.samples}. "
                f"Must be -1 (all), 0 (skip), or positive integer."
            )


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """Parsed count-based generation plan.

    Attributes:
        strategies: Mapping of canonical_name -> StrategySpec

    Properties:
        rhs_archive: RHS archive strategy (if present)
        solution_archive: Solution archive strategy (if present)
        synthetic: All non-archive strategies
    """

    strategies: Mapping[GenerationStrategyKind, StrategySpec]

    def __post_init__(self) -> None:
        """Validate plan has at least one strategy with samples > 0."""
        if not self.strategies:
            raise ValueError("At least one generation strategy must be configured")

        active_strategies = [spec for spec in self.strategies.values() if spec.samples != 0]
        if not active_strategies:
            raise ValueError("At least one strategy must have samples > 0 or samples = -1")

    @property
    def rhs_archive(self) -> StrategySpec | None:
        """Get RHS archive strategy if present."""
        return self.strategies.get(GenerationStrategyKind.RHS_ARCHIVE)

    @property
    def solution_archive(self) -> StrategySpec | None:
        """Get solution archive strategy if present."""
        return self.strategies.get(GenerationStrategyKind.SOLUTION_ARCHIVE)

    @property
    def synthetic(self) -> Mapping[str, StrategySpec]:
        """Get all synthetic (non-archive) strategies."""
        excluded = {_RHS_ARCHIVE, _SOLUTION_ARCHIVE}
        return {name: spec for name, spec in self.strategies.items() if name not in excluded}


def _merge_specs(existing: StrategySpec, incoming: StrategySpec) -> StrategySpec:
    """Merge two specs with the same canonical name.

    Counts add unless either is -1 (all); options merge with incoming winning.
    """
    if existing.samples == -1 or incoming.samples == -1:
        merged_samples = -1
    elif existing.samples > 0 and incoming.samples > 0:
        merged_samples = existing.samples + incoming.samples
    else:
        merged_samples = max(existing.samples, incoming.samples)

    return StrategySpec(
        raw_name=incoming.raw_name,
        kind=existing.kind,
        samples=merged_samples,
        options={**existing.options, **incoming.options},
    )


def plan_from_specs(specs: Sequence[StrategySpec]) -> GenerationPlan:
    """Build a GenerationPlan from pre-validated StrategySpec objects.

    Duplicate canonical names are merged: counts add, options override.

    Args:
        specs: Sequence of strategy specs (already canonicalized).

    Returns:
        GenerationPlan with all strategies merged.

    Raises:
        ValueError: If no active strategies remain after merging.
    """
    aggregated: dict[GenerationStrategyKind, StrategySpec] = {}
    for spec in specs:
        existing = aggregated.get(spec.kind)
        aggregated[spec.kind] = _merge_specs(existing, spec) if existing else spec
    return GenerationPlan(strategies=aggregated)


def _parse_strategy_entry(
    raw_spec: Mapping[str, Any],
    index: int,
) -> StrategySpec:
    """Parse a single [[generation.strategy]] raw dict entry into a StrategySpec."""
    raw_name = raw_spec.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(
            f"'generation.strategy' entry at index {index} is missing a non-empty 'name'"
        )

    if "samples" not in raw_spec:
        raise ValueError(
            f"'generation.strategy' entry '{raw_name}' is missing 'samples' field. "
            f"Use samples=N (exact count), samples=-1 (all), or samples=0 (skip)."
        )

    try:
        samples = int(raw_spec["samples"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'generation.strategy' entry '{raw_name}' has non-integer 'samples'"
        ) from exc

    if samples < -1:
        raise ValueError(
            f"'generation.strategy' entry '{raw_name}' has invalid samples={samples}. "
            f"Must be -1 (all), 0 (skip), or positive integer."
        )

    kind = canonicalize_strategy_kind(raw_name)
    options = {k: v for k, v in raw_spec.items() if k not in {"name", "samples"}}
    return StrategySpec(raw_name=raw_name, kind=kind, samples=samples, options=options)


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise ValueError("'generation.strategy' must be an array of tables ([[generation.strategy]])")


def parse_generation_plan(generation_cfg: Mapping[str, Any]) -> GenerationPlan:
    """Parse [[generation.strategy]] entries from a raw config dict into a GenerationPlan.

    Args:
        generation_cfg: The [generation] section as a plain dict.

    Returns:
        GenerationPlan with all strategies merged.

    Raises:
        ValueError: If config is invalid or strategies are missing.

    Examples:
        >>> config = {
        ...     "strategy": [
        ...         {"name": "random", "samples": 1000},
        ...         {"name": "krylov", "samples": 500, "krylov_iters": 15},
        ...         {"name": "solution_archive", "samples": -1, "solutions_glob": "/data/*.txt"},
        ...     ]
        ... }
        >>> plan = parse_generation_plan(config)
        >>> plan.strategies.keys()
        dict_keys(['random', 'krylov', 'solution_archive'])
        >>> plan.solution_archive.samples
        -1
    """
    raw_entries = generation_cfg.get("strategy")
    if raw_entries is None:
        raise ValueError(
            "Missing '[[generation.strategy]]' entries in config. "
            "Add at least one strategy with 'name' and 'samples' fields."
        )

    entries = _ensure_sequence(raw_entries)
    if not entries:
        raise ValueError("At least one '[[generation.strategy]]' block is required")

    specs: list[StrategySpec] = []
    for index, raw_spec in enumerate(entries):
        if not isinstance(raw_spec, Mapping):
            # ValueError (not TypeError) to match the sibling raise above:
            # both are TOML config validation errors reported to the user,
            # not internal type-contract violations.
            raise ValueError(  # noqa: TRY004
                f"'generation.strategy' entry at index {index} must be a table"
            )
        specs.append(_parse_strategy_entry(raw_spec, index))

    return plan_from_specs(specs)


__all__ = [
    "GenerationPlan",
    "StrategySpec",
    "canonicalize_strategy_name",
    "parse_generation_plan",
    "plan_from_specs",
]
