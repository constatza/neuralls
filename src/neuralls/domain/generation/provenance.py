"""Provenance of a dataset build: every contributing strategy, described completely."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralls.domain.generation.counts import rounded_counts
from neuralls.domain.generation.descriptions import describe_strategy
from neuralls.domain.generation.plan import canonicalize_strategy_kind
from neuralls.domain.generation.runner import registered_config_type, rows_per_base_system
from neuralls.domain.generation.seeds import derive_strategy_seed
from neuralls.domain.generation.specs import DatasetSpec
from neuralls.domain.generation.strategy_configs import BaseStrategyConfig
from neuralls.shared.scientific_description.presentation import (
    DatasetGenerationProvenance,
    StrategyProvenance,
)

__all__ = ["describe_dataset_generation"]


def _resolved_counts(spec: DatasetSpec) -> Mapping[str, int]:
    mixture = spec.mixture
    if mixture.counts is not None:
        return mixture.counts
    if mixture.mix is not None and mixture.total is not None:
        return rounded_counts(mixture.total, mixture.mix)
    raise ValueError("A dataset spec needs explicit counts or a mix with a total")


def describe_dataset_generation(
    spec: DatasetSpec, generated_rows: int
) -> DatasetGenerationProvenance:
    """Describe every strategy that contributed rows to a dataset, using effective seeds."""
    counts = {name: count for name, count in _resolved_counts(spec).items() if count != 0}
    overrides: Mapping[str, Mapping[str, Any]] = spec.mixture.strategy_overrides or {}
    strategies: list[StrategyProvenance] = []
    for name, count in counts.items():
        options = dict(overrides.get(name, {}))
        options.setdefault("seed", derive_strategy_seed(spec.mixture.seed, name))
        options["samples"] = count
        cfg: BaseStrategyConfig = registered_config_type(name)(**options)
        if count > 0:
            rows: int | None = count * rows_per_base_system(name, options)
        else:
            rows = generated_rows if len(counts) == 1 else None
        strategies.append(
            describe_strategy(canonicalize_strategy_kind(name), cfg, sample_count=rows)
        )
    return DatasetGenerationProvenance(
        strategies=tuple(strategies), generated_rows=generated_rows, normalization=spec.normalize
    )
