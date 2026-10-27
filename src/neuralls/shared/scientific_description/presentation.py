"""Immutable presentation states for methods, datasets, and comparison cohorts.

A method moves through separate types as its presentation becomes trustworthy:
configured (standalone or dataset-qualified) -> prepared (cohort validated, compact
facts chosen) -> realized (per-system facts captured after setup). Prepared and
realized types validate their invariants on construction, so a value of those types
cannot exist around the transitions that produce them. Rendered text is never stored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import assert_never

from neuralls.shared.scientific_description.catalog import (
    SAMPLE_COUNT,
    SEED,
    FactPhase,
    FactVisibility,
    fact_definition,
)
from neuralls.shared.scientific_description.models import (
    FieldImportance,
    IntegerFact,
    ScientificDescription,
    ScientificFact,
)
from neuralls.shared.types import (
    GenerationStrategyKind,
    MatrixId,
    NormalizeType,
    PreconditionerName,
    SystemId,
)


class DuplicateMethodPresentationError(ValueError):
    """Two methods of one cohort have the same compact scientific presentation.

    ``method_ids`` are diagnostic machine identifiers only; they never become labels.
    """

    def __init__(self, method_ids: tuple[PreconditionerName, ...]) -> None:
        self.method_ids = method_ids
        super().__init__(
            "Methods have identical scientific presentations; add or expose a distinguishing "
            f"fact (method ids: {', '.join(method_ids)})"
        )


@dataclass(frozen=True, slots=True)
class EditorialPresentation:
    """An explicit, validated human title for a case or job.

    It is cosmetic: it never replaces the scientific facts of a method or dataset.
    """

    title: str

    def __post_init__(self) -> None:
        """Reject blank, multi-line, control-character, or identifier-shaped titles."""
        if not self.title or self.title != self.title.strip():
            raise ValueError("EditorialPresentation.title must be non-empty and trimmed")
        if not self.title.isprintable():
            raise ValueError("EditorialPresentation.title must be a single printable line")
        if any("_" in word for word in self.title.split()):
            raise ValueError("EditorialPresentation.title must not contain identifier text")


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
        if any(fact.key in (SAMPLE_COUNT, SEED) for fact in self.parameters):
            raise ValueError("StrategyProvenance.parameters must not hold the count or the seed")

    @property
    def facts(self) -> tuple[ScientificFact, ...]:
        """Parameters in display order with the row count and any explicit seed derived in."""
        count = IntegerFact(SAMPLE_COUNT, self.rows, FieldImportance.ESSENTIAL)
        leading = 1 if self.kind in _COUNT_AFTER_LEADING_FACT else len(self.parameters)
        match self.seed:
            case ExplicitSeed(value=value):
                seed: tuple[ScientificFact, ...] = (
                    IntegerFact(SEED, value, FieldImportance.ESSENTIAL),
                )
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


@dataclass(frozen=True, slots=True)
class ComparisonScopePresentation:
    """What a comparison ran on."""

    dataset: DatasetPresentation


type ComparisonTitle = EditorialPresentation | ComparisonScopePresentation
"""A case/comparison title: an explicit editorial name, or the dataset it ran on."""


@dataclass(frozen=True, slots=True)
class StandaloneMethodPresentation:
    """A method that does not depend on a training dataset."""

    method: ScientificDescription


@dataclass(frozen=True, slots=True)
class DatasetQualifiedMethodPresentation:
    """A dataset-trained method; the dataset is rendered once after the method."""

    method: ScientificDescription
    dataset: DatasetPresentation


type ConfiguredMethodPresentation = (
    StandaloneMethodPresentation | DatasetQualifiedMethodPresentation
)


@dataclass(frozen=True, slots=True)
class MethodPresentationCandidate:
    """One configured method awaiting cohort validation."""

    method_id: PreconditionerName
    presentation: ConfiguredMethodPresentation


@dataclass(frozen=True, slots=True)
class UnvalidatedComparisonPresentation:
    """A complete configured cohort that execution and tracking must not accept."""

    methods: tuple[MethodPresentationCandidate, ...]
    scope: ComparisonScopePresentation


def with_method(
    configured: ConfiguredMethodPresentation, method: ScientificDescription
) -> ConfiguredMethodPresentation:
    """Return ``configured`` with its method description replaced, keeping its variant."""
    return replace(configured, method=method)


def method_description(configured: ConfiguredMethodPresentation) -> ScientificDescription:
    """Return the method part, which both variants share."""
    return configured.method


def description_facts(description: ScientificDescription) -> tuple[ScientificFact, ...]:
    """Return the facts a catalog policy allows anywhere, in description order."""
    return tuple(
        fact
        for fact in (*description.facts, *description.context_facts)
        if fact_definition(fact.key).visibility_policy is not FactVisibility.NEVER
    )


@dataclass(frozen=True, slots=True)
class PreparedMethodPresentation:
    """A validated configured method with the facts chosen for compact labels."""

    method_id: PreconditionerName
    configured: ConfiguredMethodPresentation
    compact_facts: tuple[ScientificFact, ...]

    def __post_init__(self) -> None:
        """Require compact facts to be a visible subset of the configured facts."""
        available = description_facts(method_description(self.configured))
        for fact in self.compact_facts:
            if fact not in available:
                raise ValueError(
                    f"compact fact {fact.key.value!r} is not a configured fact of this method"
                )

    @property
    def detailed_facts(self) -> tuple[ScientificFact, ...]:
        """Every configured fact; derived so it cannot disagree with ``configured``."""
        return description_facts(method_description(self.configured))

    @property
    def compact(self) -> ConfiguredMethodPresentation:
        """The configured presentation restricted to compact-label facts."""
        return with_method(self.configured, self._view(self.compact_facts))

    @property
    def detailed(self) -> ConfiguredMethodPresentation:
        """The configured presentation with every configured fact."""
        return with_method(self.configured, self._view(self.detailed_facts))

    def _view(self, facts: tuple[ScientificFact, ...]) -> ScientificDescription:
        base = method_description(self.configured)
        return ScientificDescription(base_name=base.base_name, facts=facts)

    def compact_identity(self) -> tuple[object, ...]:
        """Canonical structured identity of the compact label, with no rendered text."""
        base = method_description(self.configured)
        facts = tuple((fact.key.value, fact.value) for fact in self.compact_facts)
        match self.configured:
            case StandaloneMethodPresentation():
                return (base.base_name, facts)
            case DatasetQualifiedMethodPresentation(dataset=dataset):
                return (base.base_name, facts, dataset_qualifier(dataset))
            case _:
                assert_never(self.configured)


@dataclass(frozen=True, slots=True)
class PreparedComparisonPresentation:
    """A cohort whose methods are non-empty, uniquely identified, and unambiguous."""

    methods: tuple[PreparedMethodPresentation, ...]
    scope: ComparisonScopePresentation

    def __post_init__(self) -> None:
        """Reject an empty cohort, repeated method ids, or colliding compact presentations."""
        if not self.methods:
            raise ValueError("PreparedComparisonPresentation requires at least one method")
        ids = [method.method_id for method in self.methods]
        if len(ids) != len(set(ids)):
            raise ValueError("PreparedComparisonPresentation method ids must be unique")
        by_identity: dict[tuple[object, ...], list[PreconditionerName]] = {}
        for method in self.methods:
            by_identity.setdefault(method.compact_identity(), []).append(method.method_id)
        for colliding in by_identity.values():
            if len(colliding) > 1:
                raise DuplicateMethodPresentationError(tuple(colliding))

    def method(self, method_id: PreconditionerName) -> PreparedMethodPresentation:
        """Return the prepared presentation of ``method_id``; an unknown id is a caller bug."""
        for candidate in self.methods:
            if candidate.method_id == method_id:
                return candidate
        raise KeyError(f"no prepared presentation for method {method_id!r}")


@dataclass(frozen=True, slots=True)
class SystemRealization:
    """Facts known only after setup on one system."""

    system_id: SystemId
    matrix_id: MatrixId
    facts: tuple[ScientificFact, ...]

    def __post_init__(self) -> None:
        """Allow only facts the catalog defines as known at realization time."""
        for fact in self.facts:
            if fact_definition(fact.key).phase is not FactPhase.REALIZED:
                raise ValueError(f"fact {fact.key.value!r} is not a realized fact")


@dataclass(frozen=True, slots=True)
class RealizedMethodPresentation:
    """A prepared method plus the facts captured on every system it ran."""

    prepared: PreparedMethodPresentation
    systems: tuple[SystemRealization, ...]

    def __post_init__(self) -> None:
        """Require at least one system and unique system ids."""
        if not self.systems:
            raise ValueError("RealizedMethodPresentation requires at least one system")
        ids = [system.system_id for system in self.systems]
        if len(ids) != len(set(ids)):
            raise ValueError("RealizedMethodPresentation system ids must be unique")


type ExecutedMethodPresentation = RealizedMethodPresentation | PreparedMethodPresentation
"""What a finished method workload reports from.

A method that completed setup on at least one system carries the realized state. A
method whose every setup failed has no realized facts, so it stays in the prepared
state its legend and run name were chosen from.
"""


def strategy_base_name(kind: GenerationStrategyKind) -> str:
    """Return the controlled display name of a generation strategy."""
    match kind:
        case GenerationStrategyKind.RANDOM | GenerationStrategyKind.NORMAL:
            return "Random Normal Solutions"
        case GenerationStrategyKind.KRYLOV:
            return "Krylov Subspace Solutions"
        case GenerationStrategyKind.RHS_ARCHIVE:
            return "Archived RHS Solves"
        case GenerationStrategyKind.SOLUTION_ARCHIVE:
            return "Archived Solutions"
        case GenerationStrategyKind.VALIDATED_ARCHIVE:
            return "Validated Archive Pairs"
        case GenerationStrategyKind.SCALED_SOLUTIONS:
            return "Scaled Archived Solutions"
        case GenerationStrategyKind.SPARSE_RHS:
            return "Sparse RHS Solves"
        case GenerationStrategyKind.RESIDUALS:
            return "CG Residuals"
        case GenerationStrategyKind.GAUSSIAN_RESIDUALS:
            return "Gaussian CG Residuals"
        case GenerationStrategyKind.EIGENVECTOR_FORWARD:
            return "Eigenvector Solutions"
        case GenerationStrategyKind.EIGENVECTOR_INVERSE:
            return "Eigenvector RHS Solves"
        case GenerationStrategyKind.GAUSSIAN_FORWARD:
            return "Gaussian Solutions"
        case GenerationStrategyKind.GAUSSIAN_INVERSE:
            return "Gaussian RHS Solves"
        case GenerationStrategyKind.UNIFORM_FORWARD:
            return "Uniform Solutions"
        case GenerationStrategyKind.UNIFORM_INVERSE:
            return "Uniform RHS Solves"
        case GenerationStrategyKind.CONSTANT_FORWARD:
            return "Constant Solutions"
        case GenerationStrategyKind.CONSTANT_INVERSE:
            return "Constant RHS Solves"
        case GenerationStrategyKind.NEUTRAL_ONES:
            return "Ones Solutions"
        case GenerationStrategyKind.SMOOTHER_FILTERED_PROBES:
            return "Jacobi-Filtered Probes"
        case GenerationStrategyKind.SINUSOIDAL_FIELD:
            return "Sinusoidal RHS Solves"
        case _:
            assert_never(kind)


_QUALIFIER_VISIBILITY = frozenset({FactVisibility.ALWAYS, FactVisibility.WHEN_NON_DEFAULT})


_MIXTURE_BASE_NAME = "Mixture"


def dataset_qualifier(dataset: DatasetPresentation) -> tuple[ScientificDescription, ...]:
    """Describe the dataset by its generation strategies, one description each.

    A single strategy's ``N`` is the dataset's total. When several strategies mix, a
    leading ``Mixture`` description carries the total so every dataset states how many
    samples it holds. Reproducibility facts such as seeds stay in the structured
    provenance only; a qualifier shows what generated the data, not how it was seeded.
    """
    generation = dataset.generation
    strategies = tuple(
        ScientificDescription(
            base_name=strategy_base_name(strategy.kind),
            facts=tuple(
                fact
                for fact in strategy.facts
                if fact_definition(fact.key).visibility_policy in _QUALIFIER_VISIBILITY
            ),
        )
        for strategy in generation.strategies
    )
    if len(strategies) == 1:
        return strategies
    total = IntegerFact(SAMPLE_COUNT, generation.generated_rows, FieldImportance.ESSENTIAL)
    return (ScientificDescription(_MIXTURE_BASE_NAME, facts=(total,)), *strategies)


__all__ = [
    "ComparisonScopePresentation",
    "ComparisonTitle",
    "ConfiguredMethodPresentation",
    "DatasetGenerationProvenance",
    "DatasetPresentation",
    "DatasetQualifiedMethodPresentation",
    "DatasetSourceProvenance",
    "DerivedSeed",
    "DuplicateMethodPresentationError",
    "EditorialPresentation",
    "ExecutedMethodPresentation",
    "ExplicitSeed",
    "MethodPresentationCandidate",
    "NoSeed",
    "PreparedComparisonPresentation",
    "PreparedMethodPresentation",
    "RealizedMethodPresentation",
    "StandaloneMethodPresentation",
    "StrategyProvenance",
    "StrategySeed",
    "SystemRealization",
    "UnvalidatedComparisonPresentation",
    "dataset_qualifier",
    "description_facts",
    "method_description",
    "strategy_base_name",
    "with_method",
]
