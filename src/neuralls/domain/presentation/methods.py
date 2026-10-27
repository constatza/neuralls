"""Immutable presentation states for methods and comparison cohorts.

A method moves through separate types as its presentation becomes trustworthy:
configured (standalone or dataset-qualified) -> prepared (cohort validated, compact
facts chosen) -> executed (per-system facts captured after setup). Prepared and
realized types validate their invariants on construction, so a value of those types
cannot exist around the transitions that produce them. Rendered text is never stored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import assert_never

from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.qualifier import dataset_qualifier
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.shared.scientific_description.catalog import (
    FactPhase,
    FactVisibility,
    fact_definition,
)
from neuralls.shared.scientific_description.models import (
    DescriptionIdentity,
    ScientificDescription,
    ScientificFact,
    require_unique_keys,
)
from neuralls.shared.types import MatrixId, PreconditionerName, SystemId


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


def description_facts(description: ScientificDescription) -> tuple[ScientificFact, ...]:
    """Return the facts a catalog policy allows anywhere, in description order."""
    return tuple(
        fact
        for fact in (*description.facts, *description.context_facts)
        if fact_definition(fact.key).visibility_policy is not FactVisibility.NEVER
    )


@dataclass(frozen=True, slots=True)
class CompactIdentity:
    """What makes a prepared method's compact label unique within a cohort.

    ``dataset`` is empty for a standalone method; a dataset-qualified method always has at
    least one qualifier description, so the two cannot be confused.
    """

    method: DescriptionIdentity
    dataset: tuple[DescriptionIdentity, ...]


@dataclass(frozen=True, slots=True)
class PreparedMethodPresentation:
    """A validated configured method with the facts chosen for compact labels."""

    method_id: PreconditionerName
    configured: ConfiguredMethodPresentation
    compact_facts: tuple[ScientificFact, ...]

    def __post_init__(self) -> None:
        """Require unique compact facts forming an ordered subsequence of the configured facts."""
        require_unique_keys(self.compact_facts, "PreparedMethodPresentation.compact_facts")
        remaining = iter(description_facts(self.configured.method))
        for fact in self.compact_facts:
            if not any(fact == configured for configured in remaining):
                raise ValueError(
                    f"compact fact {fact.key.value!r} is not a configured fact of this method, "
                    "or is out of its configured order"
                )

    @property
    def detailed_facts(self) -> tuple[ScientificFact, ...]:
        """Every configured fact; derived so it cannot disagree with ``configured``."""
        return description_facts(self.configured.method)

    @property
    def compact(self) -> ConfiguredMethodPresentation:
        """The configured presentation restricted to compact-label facts."""
        return replace(self.configured, method=self._view(self.compact_facts))

    @property
    def detailed(self) -> ConfiguredMethodPresentation:
        """The configured presentation with every configured fact."""
        return replace(self.configured, method=self._view(self.detailed_facts))

    def _view(self, facts: tuple[ScientificFact, ...]) -> ScientificDescription:
        return ScientificDescription(base_name=self.configured.method.base_name, facts=facts)

    def compact_identity(self) -> CompactIdentity:
        """Canonical structured identity of the compact label, with no rendered text."""
        method = self._view(self.compact_facts).identity
        match self.configured:
            case StandaloneMethodPresentation():
                return CompactIdentity(method=method, dataset=())
            case DatasetQualifiedMethodPresentation(dataset=dataset):
                return CompactIdentity(
                    method=method,
                    dataset=tuple(part.identity for part in dataset_qualifier(dataset)),
                )
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
        by_identity: dict[CompactIdentity, list[PreconditionerName]] = {}
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
        """Allow only unique facts the catalog defines as known at realization time."""
        require_unique_keys(self.facts, "SystemRealization")
        for fact in self.facts:
            if fact_definition(fact.key).phase is not FactPhase.REALIZED:
                raise ValueError(f"fact {fact.key.value!r} is not a realized fact")


@dataclass(frozen=True, slots=True)
class ExecutedMethod:
    """A prepared method plus the facts captured on the systems whose setup succeeded.

    ``systems`` is empty when no setup succeeded (the method is then reported from its
    prepared state alone); otherwise it holds one realization per system, unique by id.
    """

    prepared: PreparedMethodPresentation
    systems: tuple[SystemRealization, ...]

    def __post_init__(self) -> None:
        """Require unique system ids."""
        ids = [system.system_id for system in self.systems]
        if len(ids) != len(set(ids)):
            raise ValueError("ExecutedMethod system ids must be unique")
