"""The prepared state of a comparison batch: decided outcomes and ready executions."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import assert_never

from loguru import logger

from neuralls.composition.assignments.comparison_resolution import (
    ResolvedComparisonSpecs,
)
from neuralls.composition.assignments.comparison_topology import ComparisonTopology
from neuralls.composition.comparison.models import (
    ComparisonCompleted,
    ComparisonEntryFailed,
    ComparisonOutcome,
    ComparisonPresentation,
    ComparisonReused,
    PlannedSystem,
    is_comparison_outcome,
)
from neuralls.composition.tracking.run_specs import (
    comparison_run,
)
from neuralls.domain.benchmarking.selection import (
    NoReferenceConfigured,
    ReferenceSelection,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import (
    PreparedComparisonPresentation,
)
from neuralls.domain.presentation.run_names import NamedRun, Untimed
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.experiments import (
    ComparisonRegistryEntry,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.run_names import run_name_text


def described_failure(prepared: PreparedComparisonExecution, exc: Exception) -> ComparisonOutcome:
    """The outcome of a described comparison whose execution raised."""
    run = prepared.run_name
    logger.error("{} failed: {}", run_name_text(run), exc)
    return ComparisonEntryFailed(comparison_id=prepared.entry.id, run=run, reason=str(exc))


@dataclass(frozen=True, slots=True)
class PreparedComparisonExecution:
    """One entry, resolved and confirmed to need real execution.

    `cleanup` owns the entry's artifact-lease lifetime (already resolved into
    `resolved.specs`' baked-in checkpoint paths) — the caller must enter it
    into its own ExitStack (or close it directly) once execution is done.

    The presentation must describe exactly the resolved methods, in order, and a title
    composed from the dataset scope must be that presentation's own scope.
    """

    cfg: ComparisonConfig
    planned: PlannedSystem
    topology: ComparisonTopology
    cleanup: contextlib.ExitStack
    resolved: ResolvedComparisonSpecs
    identity: StageIdentity
    settings: NeurallsSettings
    presentation: PreparedComparisonPresentation
    """The validated scientific presentation of every method; execution requires it."""
    title: ComparisonTitle
    reference_selection: ReferenceSelection = field(default_factory=NoReferenceConfigured)

    def __post_init__(self) -> None:
        """Require the presentation and title to describe this execution's own methods."""
        spec_names = tuple(spec.name for spec in self.resolved.specs)
        method_ids = tuple(method.method_id for method in self.presentation.methods)
        if method_ids != spec_names:
            raise ValueError(
                f"presentation methods {method_ids!r} differ from the resolved specs "
                f"{spec_names!r}; they must be the same methods in the same order"
            )
        match self.title:
            case ComparisonScopePresentation() as scope:
                if scope != self.presentation.scope:
                    raise ValueError("a scope title must be the presentation's own scope")
            case EditorialPresentation():
                pass
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def entry(self) -> ComparisonRegistryEntry:
        """The planned system's concrete registry entry."""
        return self.planned.entry

    @property
    def run_name(self) -> NamedRun:
        """The typed name of this system's comparison run, as MLflow names it."""
        return comparison_run(
            self.planned,
            self.title,
            self.presentation.scope.dataset,
            self.cfg.notation_policy,
            Untimed(),
        )

    @property
    def comparison_presentation(self) -> ComparisonPresentation:
        """The validated cohort, its title, and the case's notation, for the figures."""
        return ComparisonPresentation(
            methods=self.presentation, title=self.title, notation=self.cfg.notation_policy
        )


type PendingComparison = ComparisonOutcome | PreparedComparisonExecution
"""A planned system's state before execution: already decided, or ready to execute."""


def _planned_system_matches(item: PendingComparison, planned: PlannedSystem) -> bool:
    """Whether ``item`` is the state of ``planned`` and not of some other system."""
    match item:
        case PreparedComparisonExecution():
            return item.planned is planned
        case ComparisonCompleted() | ComparisonReused() | ComparisonEntryFailed():
            return item.comparison_id == planned.entry.id
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class PreparedComparisonBatch:
    """Every comparison of a case, resolved and presentation-validated; nothing has run.

    Session tracking and execution accept only this type, so an unvalidated cohort cannot
    reach MLflow or a numerical kernel. A comparison that is a reuse hit or a failure
    carries its outcome; one that must execute carries its prepared execution. Item ``i``
    is the state of planned system ``i``.
    """

    items: tuple[PendingComparison, ...]
    planned: tuple[PlannedSystem, ...]

    def __post_init__(self) -> None:
        """Require item ``i`` to be the state of planned system ``i``."""
        if len(self.items) != len(self.planned):
            raise ValueError("A prepared batch needs exactly one item per planned system")
        for position, (item, planned) in enumerate(zip(self.items, self.planned, strict=True)):
            if not _planned_system_matches(item, planned):
                raise ValueError(
                    f"A prepared batch item {position} is not the state of its planned system "
                    f"{planned.entry.id!r}"
                )


@dataclass(frozen=True, slots=True)
class PendingPartition:
    """Pending comparisons split into those ready to execute and those already decided."""

    executions: tuple[PreparedComparisonExecution, ...]
    decided: tuple[ComparisonOutcome, ...]


def partition_pending(pending: Sequence[PendingComparison]) -> PendingPartition:
    """Split ``pending`` by the state of each item, preserving order within each side."""
    executions: list[PreparedComparisonExecution] = []
    decided: list[ComparisonOutcome] = []
    for item in pending:
        if is_comparison_outcome(item):
            decided.append(item)
        else:
            executions.append(item)
    return PendingPartition(executions=tuple(executions), decided=tuple(decided))
