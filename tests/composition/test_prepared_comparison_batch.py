"""A prepared batch proves that its items, plan, and presentations describe the same systems."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from neuralls.composition.assignments.comparison_batch import (
    PreparedComparisonBatch,
    PreparedComparisonExecution,
    ResolvedComparisonSpecs,
)
from neuralls.composition.comparison.models import (
    ComparisonReused,
    MatrixRef,
    PlannedSystem,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import (
    MethodPresentationCandidate,
    PreparedComparisonPresentation,
    StandaloneMethodPresentation,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import DatasetPresentation, DatasetSourceProvenance
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.presentation.titles import ComparisonScopePresentation, EditorialPresentation
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.types import MatrixId, PreconditionerName

type PlannedFactory = Callable[[str], PlannedSystem]
type ExecutionFactory = Callable[[PlannedSystem], PreparedComparisonExecution]


@pytest.fixture
def planned_system() -> PlannedFactory:
    def build(entry_id: str) -> PlannedSystem:
        return PlannedSystem(
            entry=ComparisonRegistryEntry(
                id=entry_id, matrix_dataset="ds", rhs_source={"kind": "gaussian"}
            ),
            matrix=MatrixRef(matrix_id=MatrixId(0), row=0),
            rhs_row=None,
            rhs_ordinal=0,
            provenance=SharedSingleMatrix(raw_matrix_index=0),
        )

    return build


@pytest.fixture
def jacobi_specs() -> list[PreconditionerConfig]:
    return [StandardPreconditionerConfig(type=PreconditionerType.JACOBI)]


@pytest.fixture
def execution(
    prepared_presentation: PreparedComparisonPresentation,
    jacobi_specs: list[PreconditionerConfig],
) -> ExecutionFactory:
    def build(planned: PlannedSystem) -> PreparedComparisonExecution:
        return PreparedComparisonExecution(
            cfg=MagicMock(),
            planned=planned,
            topology=MagicMock(),
            cleanup=contextlib.ExitStack(),
            resolved=ResolvedComparisonSpecs(
                specs=jacobi_specs, warnings=(), checkpoint_digests={}
            ),
            identity=StageIdentity.build(
                "comparison", {"entry": canonical_digest(planned.entry.id)}
            ),
            settings=MagicMock(),
            presentation=prepared_presentation,
            title=prepared_presentation.scope,
        )

    return build


def test_batch_accepts_executions_and_outcomes_matching_the_plan(
    planned_system: PlannedFactory, execution: ExecutionFactory, comparison_run: NamedRun
) -> None:
    first, second = planned_system("a"), planned_system("b")
    reused = ComparisonReused(comparison_id="b", run=comparison_run)

    batch = PreparedComparisonBatch(items=(execution(first), reused), planned=(first, second))

    assert batch.planned == (first, second)


def test_batch_rejects_an_execution_of_a_different_planned_system(
    planned_system: PlannedFactory, execution: ExecutionFactory
) -> None:
    first, second = planned_system("a"), planned_system("b")

    with pytest.raises(ValueError, match="planned system"):
        PreparedComparisonBatch(
            items=(execution(second), execution(first)), planned=(first, second)
        )


def test_batch_rejects_an_outcome_of_a_different_comparison(
    planned_system: PlannedFactory, comparison_run: NamedRun
) -> None:
    first = planned_system("a")
    other = ComparisonReused(comparison_id="zzz", run=comparison_run)

    with pytest.raises(ValueError, match="planned system"):
        PreparedComparisonBatch(items=(other,), planned=(first,))


def test_batch_is_slotted(planned_system: PlannedFactory) -> None:
    batch = PreparedComparisonBatch(items=(), planned=())

    assert not hasattr(batch, "__dict__")


def test_execution_rejects_a_presentation_whose_methods_differ_from_the_resolved_specs(
    planned_system: PlannedFactory,
    execution: ExecutionFactory,
) -> None:
    prepared = execution(planned_system("a"))
    identity_only = [StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)]

    with pytest.raises(ValueError, match="resolved specs"):
        replace(prepared, resolved=replace(prepared.resolved, specs=identity_only))


def test_execution_rejects_specs_in_a_different_order_than_the_presentation(
    planned_system: PlannedFactory,
    execution: ExecutionFactory,
    jacobi_specs: list[PreconditionerConfig],
    dataset_presentation: DatasetPresentation,
) -> None:
    del jacobi_specs
    cohort = prepare_presentations(
        UnvalidatedComparisonPresentation(
            methods=(
                MethodPresentationCandidate(
                    PreconditionerName("jacobi"),
                    StandaloneMethodPresentation(ScientificDescription("Jacobi")),
                ),
                MethodPresentationCandidate(
                    PreconditionerName("identity"),
                    StandaloneMethodPresentation(ScientificDescription("Identity")),
                ),
            ),
            scope=ComparisonScopePresentation(dataset_presentation),
        )
    )
    base = execution(planned_system("a"))
    swapped = [
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
    ]

    with pytest.raises(ValueError, match="resolved specs"):
        replace(base, presentation=cohort, resolved=replace(base.resolved, specs=swapped))


def test_execution_rejects_a_scope_title_for_a_different_dataset(
    planned_system: PlannedFactory,
    execution: ExecutionFactory,
    dataset_presentation: DatasetPresentation,
) -> None:
    base = execution(planned_system("a"))
    other_dataset = replace(
        dataset_presentation,
        source=DatasetSourceProvenance(ScientificDescription("Other problem")),
    )

    with pytest.raises(ValueError, match="title"):
        replace(base, title=ComparisonScopePresentation(other_dataset))


def test_execution_accepts_an_editorial_title_regardless_of_scope(
    planned_system: PlannedFactory, execution: ExecutionFactory
) -> None:
    base = execution(planned_system("a"))

    editorial = EditorialPresentation("Spectral study")

    assert replace(base, title=editorial).title == editorial
