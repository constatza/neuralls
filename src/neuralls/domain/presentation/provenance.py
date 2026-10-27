"""Scientific provenance of a generated dataset: its source, strategies, rows, and seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from neuralls.shared.scientific_description.catalog import SAMPLE_COUNT, SEED, make_fact
from neuralls.shared.scientific_description.models import (
    ScientificDescription,
    ScientificFact,
    require_unique_keys,
)
from neuralls.shared.types import GenerationStrategyKind, NormalizeType


@dataclass(frozen=True, slots=True)
class DatasetSourceProvenance:
    """The problem/matrix family a dataset was generated from."""

    source: ScientificDescription


@dataclass(frozen=True, slots=True)
class ExplicitSeed:
    """A seed the configuration fixed for one strategy; every binding's stream used it."""

    value: int


@dataclass(frozen=True, slots=True)
class DerivedSeed:
    """The strategy took no seed of its own: each binding's stream derives from the root seed."""


@dataclass(frozen=True, slots=True)
class NoSeed:
    """The configuration explicitly unseeded the strategy, so its streams are not reproducible."""


type StrategySeed = ExplicitSeed | DerivedSeed | NoSeed

_COUNT_AFTER_LEADING_FACT = frozenset(
    {
        GenerationStrategyKind.RESIDUALS,
        GenerationStrategyKind.GAUSSIAN_RESIDUALS,
        GenerationStrategyKind.SMOOTHER_FILTERED_PROBES,
    }
)
"""Trajectory strategies read "k=20, N=5000, ..." (the cap, the count, the rest); every
other strategy lists its parameters and then the count."""


@dataclass(frozen=True, slots=True)
class StrategyProvenance:
    """One generation strategy: the rows it produced, its parameters, and its seed.

    ``rows`` is the single record of the strategy's sample count; the ``N`` fact is
    derived from it in ``facts`` and is never stored. ``parameters`` hold only the
    algorithmic facts, so a count or seed cannot be stored twice or disagree.
    """

    kind: GenerationStrategyKind
    rows: int
    parameters: tuple[ScientificFact, ...]
    seed: StrategySeed

    def __post_init__(self) -> None:
        """Require the strategy to have produced rows and to carry no count/seed fact."""
        if self.rows < 1:
            raise ValueError(f"StrategyProvenance.rows must be positive, got {self.rows}")
        require_unique_keys(self.parameters, "StrategyProvenance.parameters")
        if any(fact.key in (SAMPLE_COUNT, SEED) for fact in self.parameters):
            raise ValueError("StrategyProvenance.parameters must not hold the count or the seed")

    @property
    def facts(self) -> tuple[ScientificFact, ...]:
        """Parameters in display order with the row count and any explicit seed derived in."""
        count = make_fact(SAMPLE_COUNT, self.rows)
        leading = 1 if self.kind in _COUNT_AFTER_LEADING_FACT else len(self.parameters)
        match self.seed:
            case ExplicitSeed(value=value):
                seed: tuple[ScientificFact, ...] = (make_fact(SEED, value),)
            case DerivedSeed() | NoSeed():
                seed = ()
            case _:
                assert_never(self.seed)
        return (*self.parameters[:leading], count, *self.parameters[leading:], *seed)


@dataclass(frozen=True, slots=True)
class DatasetGenerationProvenance:
    """Everything that distinguishes how a dataset's samples were generated.

    ``root_seed`` is the mixture seed that binding allocation and every
    ``DerivedSeed`` strategy stream derive from; it is not itself a stream's seed.
    """

    strategies: tuple[StrategyProvenance, ...]
    generated_rows: int
    normalization: NormalizeType
    root_seed: int

    def __post_init__(self) -> None:
        """Require unique strategies whose rows add up to the dataset's rows."""
        if not self.strategies:
            raise ValueError("DatasetGenerationProvenance requires at least one strategy")
        kinds = [strategy.kind for strategy in self.strategies]
        if len(set(kinds)) != len(kinds):
            raise ValueError("DatasetGenerationProvenance strategy kinds must be unique")
        total = sum(strategy.rows for strategy in self.strategies)
        if total != self.generated_rows:
            raise ValueError(
                f"Strategy rows sum to {total} but the dataset has {self.generated_rows} rows"
            )


@dataclass(frozen=True, slots=True)
class DatasetPresentation:
    """A dataset's structured scientific identity, independent of any rendering."""

    source: DatasetSourceProvenance
    generation: DatasetGenerationProvenance
