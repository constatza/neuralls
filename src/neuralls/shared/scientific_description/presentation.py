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
    FactPhase,
    FactVisibility,
    fact_definition,
)
from neuralls.shared.scientific_description.models import (
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
class StrategyProvenance:
    """One generation strategy and the scientific facts that parameterize it."""

    kind: GenerationStrategyKind
    facts: tuple[ScientificFact, ...]


@dataclass(frozen=True, slots=True)
class DatasetGenerationProvenance:
    """Everything that distinguishes how a dataset's samples were generated."""

    strategies: tuple[StrategyProvenance, ...]
    generated_rows: int
    normalization: NormalizeType

    def __post_init__(self) -> None:
        """Require at least one strategy and a positive row count."""
        if not self.strategies:
            raise ValueError("DatasetGenerationProvenance requires at least one strategy")
        if self.generated_rows < 1:
            raise ValueError("DatasetGenerationProvenance.generated_rows must be positive")


@dataclass(frozen=True, slots=True)
class DatasetPresentation:
    """A dataset's structured scientific identity, independent of any rendering."""

    source: DatasetSourceProvenance
    generation: DatasetGenerationProvenance


@dataclass(frozen=True, slots=True)
class ComparisonScopePresentation:
    """What a comparison ran on."""

    dataset: DatasetPresentation


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
                return (base.base_name, facts, dataset)
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


def strategy_base_name(kind: GenerationStrategyKind) -> str:
    """Return the controlled display name of a generation strategy."""
    match kind:
        case GenerationStrategyKind.RANDOM | GenerationStrategyKind.NORMAL:
            return "Gaussian LHS"
        case GenerationStrategyKind.KRYLOV:
            return "Krylov Combinations"
        case GenerationStrategyKind.RHS_ARCHIVE:
            return "RHS Archive"
        case GenerationStrategyKind.SOLUTION_ARCHIVE:
            return "Solution Archive"
        case GenerationStrategyKind.VALIDATED_ARCHIVE:
            return "Validated Archive"
        case GenerationStrategyKind.SCALED_SOLUTIONS:
            return "Scaled Solutions"
        case GenerationStrategyKind.SPARSE_RHS:
            return "Sparse RHS"
        case GenerationStrategyKind.RESIDUALS:
            return "CG Residuals"
        case GenerationStrategyKind.GAUSSIAN_RESIDUALS:
            return "Gaussian CG Residuals"
        case GenerationStrategyKind.EIGENVECTOR_FORWARD:
            return "Eigenvector Forward"
        case GenerationStrategyKind.EIGENVECTOR_INVERSE:
            return "Eigenvector Inverse"
        case GenerationStrategyKind.GAUSSIAN_FORWARD:
            return "Gaussian Forward"
        case GenerationStrategyKind.GAUSSIAN_INVERSE:
            return "Gaussian Inverse"
        case GenerationStrategyKind.UNIFORM_FORWARD:
            return "Uniform Forward"
        case GenerationStrategyKind.UNIFORM_INVERSE:
            return "Uniform Inverse"
        case GenerationStrategyKind.CONSTANT_FORWARD:
            return "Constant Forward"
        case GenerationStrategyKind.CONSTANT_INVERSE:
            return "Constant Inverse"
        case GenerationStrategyKind.NEUTRAL_ONES:
            return "Neutral Ones"
        case GenerationStrategyKind.SMOOTHER_FILTERED_PROBES:
            return "Smoother-Filtered Probes"
        case GenerationStrategyKind.SINUSOIDAL_FIELD:
            return "Sinusoidal Field"
        case _:
            assert_never(kind)


def dataset_qualifier(dataset: DatasetPresentation) -> tuple[ScientificDescription, ...]:
    """Describe the dataset by its generation strategies, one description each."""
    return tuple(
        ScientificDescription(base_name=strategy_base_name(strategy.kind), facts=strategy.facts)
        for strategy in dataset.generation.strategies
    )


__all__ = [
    "ComparisonScopePresentation",
    "ConfiguredMethodPresentation",
    "DatasetGenerationProvenance",
    "DatasetPresentation",
    "DatasetQualifiedMethodPresentation",
    "DatasetSourceProvenance",
    "DuplicateMethodPresentationError",
    "EditorialPresentation",
    "MethodPresentationCandidate",
    "PreparedComparisonPresentation",
    "PreparedMethodPresentation",
    "RealizedMethodPresentation",
    "StandaloneMethodPresentation",
    "StrategyProvenance",
    "SystemRealization",
    "UnvalidatedComparisonPresentation",
    "dataset_qualifier",
    "description_facts",
    "method_description",
    "strategy_base_name",
    "with_method",
]
