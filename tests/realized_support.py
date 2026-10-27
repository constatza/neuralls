"""Shared builders of completed method workloads whose systems carry realized facts."""

from __future__ import annotations

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    PreparedMethodPresentation,
    SystemRealization,
)
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.shared.rhs_recipe import RhsRecipe
from neuralls.shared.scientific_description.catalog import REALIZED_LEVELS, make_fact
from neuralls.shared.scientific_description.models import ScientificFact
from neuralls.shared.types import DatasetId, MatrixId, SetupEventId, SystemId
from tests.recipe_support import GAUSSIAN


def levels(depth: int) -> tuple[ScientificFact, ...]:
    return (make_fact(REALIZED_LEVELS, depth),)


def completed_workload(
    prepared: PreparedMethodPresentation,
    depth_by_matrix: dict[int, int],
    *,
    recipe_by_matrix: dict[int, RhsRecipe] | None = None,
    realized_matrices: frozenset[int] | None = None,
) -> CompletedMethodWorkload:
    """A completed workload with one system per matrix, realized at the given depths.

    ``recipe_by_matrix`` names each matrix's recipe (default the standard normal) and
    ``realized_matrices`` limits which matrices have a realization (default all).
    """
    name = prepared.method_id
    recipes = recipe_by_matrix or {}
    tags = tuple(
        SystemTag(
            DatasetId("ds"),
            recipes.get(m, GAUSSIAN),
            MatrixId(m),
            0,
            SystemId(f"{name}-{m}"),
        )
        for m in depth_by_matrix
    )
    events = tuple(
        MatrixSetupEvent(
            SetupEventId(f"{name}:ds:{tag.matrix_id}"),
            tag.dataset,
            tag.matrix_id,
            name,
            1.0,
            (1.0,),
            MeasuredPeakMemory(1),
            SetupCompleted(),
        )
        for tag in tags
    )
    attempts = tuple(
        SolveAttempt(
            event.event_id,
            tag.system_id,
            tag.dataset,
            tag.recipe,
            tag.matrix_id,
            tag.rhs_ordinal,
            name,
            1.0,
            (1.0,),
            MeasuredPeakMemory(1),
            SolveConverged(1, 1e-8),
        )
        for tag, event in zip(tags, events, strict=True)
    )
    realized = ExecutedMethod(
        prepared,
        tuple(
            SystemRealization(
                tag.system_id, tag.matrix_id, levels(depth_by_matrix[int(tag.matrix_id)])
            )
            for tag in tags
            if realized_matrices is None or int(tag.matrix_id) in realized_matrices
        ),
    )
    return CompletedMethodWorkload(
        method=name,
        planned=tags,
        setup_events=events,
        solve_attempts=attempts,
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(1),
        config_digest="sha256:" + "0" * 64,
        method_config={"name": str(name)},
        presentation=realized,
        plot_styles={str(name): PreconditionerPlotStyle(prepared, PreconditionerType.JACOBI)},
    )
