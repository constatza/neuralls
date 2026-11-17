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
from neuralls.domain.presentation.strategy_names import snapshot_source
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.shared.scientific_description.catalog import (
    SAMPLE_COUNT,
    FactPhase,
    FactVisibility,
    fact_definition,
    make_fact,
)
from neuralls.shared.scientific_description.models import (
    DescriptionIdentity,
    ScientificDescription,
    ScientificFact,
    require_unique_keys,
)
from neuralls.shared.types import MatrixId, PreconditionerFamily, PreconditionerName, SystemId


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


class MissingMethodPresentationError(LookupError):
    """A figure or table needs a method's presentation, and none was supplied.

    The method id is a diagnostic machine identifier only; it never becomes a label.
    """

    def __init__(self, method_id: str) -> None:
        self.method_id = method_id
        super().__init__(f"no scientific presentation for method {method_id!r}")


@dataclass(frozen=True, slots=True)
class StandaloneMethodPresentation:
    """A method that does not depend on a training dataset."""

    method: ScientificDescription


@dataclass(frozen=True, slots=True)
class DatasetQualifiedMethodPresentation:
    """A dataset-trained method; the dataset is rendered once after the method.

    ``fit_rows`` counts the dataset rows the method's fit read: the folds of the pinned
    split it was fit on, which is fewer than the dataset's generated rows whenever the
    split holds rows out.
    """

    method: ScientificDescription
    dataset: DatasetPresentation
    fit_rows: int

    def __post_init__(self) -> None:
        """Require a fit that read at least one and at most every generated row."""
        if not 0 < self.fit_rows <= self.dataset.generation.generated_rows:
            raise ValueError(
                f"fit_rows must lie in [1, {self.dataset.generation.generated_rows}], "
                f"got {self.fit_rows}"
            )


def pod_source_name(dataset: DatasetPresentation) -> str | None:
    """The name of a POD method fit on ``dataset`` (POD-CG, POD-JP), if its data names one.

    Only fit data of one snapshot strategy names the method; a mixture, or a strategy that
    is not a snapshot source, leaves the plain POD name.
    """
    match dataset.generation.strategies:
        case (single,):
            source = snapshot_source(single.kind)
            if source is not None:
                return f"{PreconditionerFamily.POD.abbreviation()}-{source}"
    return None


def qualify_method(
    method: ScientificDescription, dataset: DatasetPresentation, fit_rows: int
) -> DatasetQualifiedMethodPresentation:
    """Qualify a dataset-trained method by its data; a POD method names its snapshot source.

    POD-CG (truncated-CG errors) and POD-JP (Jacobi-filtered probes) are distinct methods,
    so the source belongs in the method name, and the renderer then states the strategy's
    facts inside the method's own parenthesis instead of after a dataset qualifier.
    ``fit_rows`` is the number of dataset rows the fit read.
    """
    name = pod_source_name(dataset)
    if name is not None and method.base_name == PreconditionerFamily.POD.abbreviation():
        method = replace(method, base_name=name)
    return DatasetQualifiedMethodPresentation(method, dataset, fit_rows)


type ConfiguredMethodPresentation = (
    StandaloneMethodPresentation | DatasetQualifiedMethodPresentation
)


def method_parts(configured: ConfiguredMethodPresentation) -> tuple[ScientificDescription, ...]:
    """The descriptions a method's label is made of, method first.

    The one owner of a label's parts: the renderer draws them and the cohort identity is
    built from them, so two methods share an identity exactly when their labels agree.
    A POD method named by its snapshot source (POD-CG, POD-JP) already says which strategy
    made its data, so that strategy's facts join the method's own parenthesis
    (``POD-CG (c=100, k=20, N=4500)``) instead of following a dataset qualifier. There
    ``N`` is the method's own: the columns of the snapshot matrix it fit, one per fit
    row (a trajectory's kept steps are separate rows), not the rows generation produced.
    """
    match configured:
        case StandaloneMethodPresentation(method=method):
            return (method,)
        case DatasetQualifiedMethodPresentation(method=method, dataset=dataset, fit_rows=fit_rows):
            qualifier = dataset_qualifier(dataset)
            match qualifier:
                case (strategy,) if method.base_name == pod_source_name(dataset):
                    fitted = make_fact(SAMPLE_COUNT, fit_rows)
                    strategy_facts = tuple(
                        fitted if fact.key == SAMPLE_COUNT else fact for fact in strategy.facts
                    )
                    folded = ScientificDescription(
                        base_name=method.base_name,
                        facts=(*method.facts, *strategy_facts),
                        context_facts=(*method.context_facts, *strategy.context_facts),
                    )
                    return (folded,)
            return (method, *qualifier)
        case _:
            assert_never(configured)


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

    ``parts`` are the identities of the compact label's parts (``method_parts``), so the
    identity holds exactly what the label renders: a POD fold's fitted ``N`` and not the
    generated rows it replaces.
    """

    parts: tuple[DescriptionIdentity, ...]


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
        return CompactIdentity(tuple(part.identity for part in method_parts(self.compact)))


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
        """Return the prepared presentation of ``method_id``.

        Raises:
            MissingMethodPresentationError: When ``method_id`` is not in this cohort.
        """
        for candidate in self.methods:
            if candidate.method_id == method_id:
                return candidate
        raise MissingMethodPresentationError(method_id)


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
