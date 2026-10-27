"""Provenance of a dataset build: every contributing strategy, described completely."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neuralls.domain.generation.descriptions import describe_strategy
from neuralls.domain.generation.plan import canonicalize_strategy_kind
from neuralls.domain.generation.runner import registered_config_type
from neuralls.domain.generation.sample_writer import StrategyRowCount
from neuralls.domain.generation.specs import DatasetSpec
from neuralls.domain.presentation.provenance import (
    DatasetGenerationProvenance,
    DerivedSeed,
    ExplicitSeed,
    NoSeed,
    StrategyProvenance,
    StrategySeed,
)

__all__ = ["describe_dataset_generation"]

_SEED_OPTION = "seed"
_SOURCE_LOCATION_FIELDS = ("solutions_glob", "rhs_glob")
"""Config fields that only say where archive files live. Rows fed from a per-binding
solution or RHS file have no glob, and the location is never a described fact."""


def _strategy_seed(options: Mapping[str, Any]) -> StrategySeed:
    """The seed generation gave this strategy's config, as ``_generate_strategy_rows`` does.

    A seed present in the options is used as given, ``None`` included; only an absent seed
    is derived per binding from the mixture seed.
    """
    if _SEED_OPTION not in options:
        return DerivedSeed()
    value = options[_SEED_OPTION]
    return NoSeed() if value is None else ExplicitSeed(int(value))


def describe_dataset_generation(
    spec: DatasetSpec, realized: Sequence[StrategyRowCount]
) -> DatasetGenerationProvenance:
    """Describe every strategy that wrote rows, from the rows the writer recorded.

    Counts come from ``realized`` only: the spec's budgets are neither row counts (trace
    strategies budget rows, archives may be open-ended) nor split between strategies.

    Args:
        spec: The spec the dataset was generated from; supplies strategy options, the seed
            root and the normalization.
        realized: Rows each strategy wrote, from the sample writer.

    Returns:
        The dataset's generation provenance; its rows are the sum of ``realized``.
    """
    overrides: Mapping[str, Mapping[str, Any]] = spec.mixture.strategy_overrides or {}
    strategies: list[StrategyProvenance] = []
    for counted in realized:
        options = dict(overrides.get(counted.strategy_name, {}))
        options["samples"] = counted.rows
        config_type = registered_config_type(counted.strategy_name)
        for field in _SOURCE_LOCATION_FIELDS:
            if field in config_type.model_fields and config_type.model_fields[field].is_required():
                options.setdefault(field, "")
        config = config_type(**options)
        strategies.append(
            describe_strategy(
                canonicalize_strategy_kind(counted.strategy_name),
                config,
                rows=counted.rows,
                seed=_strategy_seed(options),
            )
        )
    return DatasetGenerationProvenance(
        strategies=tuple(strategies),
        generated_rows=sum(counted.rows for counted in realized),
        normalization=spec.normalize,
        root_seed=spec.mixture.seed,
    )
