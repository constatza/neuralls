"""An unbalanced multi-method workload with every kind of system outcome, as domain events.

Matrix 1 has three converged systems, matrix 2 one converged, one unconverged and one that
raised in the solver, matrix 3 failed its setup for ``amg`` and matrix 4 was never reached.
``jacobi`` converges everywhere; ``broken`` never gets past setup.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    MeasuredResidual,
    ResidualMeasurement,
    SetupCompleted,
    SetupFailed,
    SetupOutcome,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SolveOutcome,
    SystemTag,
    UnmeasuredResidual,
)
from neuralls.domain.solver.reporting.event_frames import (
    CONVERGENCE_SCHEMA,
    build_planned_systems_frame,
    build_setup_frame,
    build_solve_frame,
)
from neuralls.domain.solver.reporting.observations import (
    OBSERVATION_SCHEMA,
    MethodFrames,
    Outcome,
    observations_of_methods,
)
from neuralls.shared.device import SingleRun
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId
from tests.measurement_support import NO_REFERENCE, NO_RESIDUAL
from tests.provenance_support import provenance_for
from tests.recipe_support import GAUSSIAN, SINE

DATASET = DatasetId("ds")
AMG = PreconditionerName("amg")
JACOBI = PreconditionerName("jacobi")
BROKEN = PreconditionerName("broken")

RHS_PER_MATRIX = {1: 3, 2: 3, 3: 2, 4: 1}
PLANNED = tuple(
    SystemTag(DATASET, GAUSSIAN, MatrixId(matrix), ordinal, SystemId(f"m{matrix}-{ordinal}"))
    for matrix, count in RHS_PER_MATRIX.items()
    for ordinal in range(count)
)

RHS_NORM = 2.0
STOPPING_THRESHOLD_REL = 1e-6
"""``max(rtol, atol / RHS_NORM)`` for the planned systems' ``rtol = 1e-6`` and ``atol = 1e-10``."""
CONVERGED_TRUE_RESIDUAL = STOPPING_THRESHOLD_REL / 2
UNCONVERGED_TRUE_RESIDUAL = 1e-3

_CONVERGED_RESIDUAL = MeasuredResidual(rhs_norm=RHS_NORM, true_residual_rel=CONVERGED_TRUE_RESIDUAL)
_UNCONVERGED_RESIDUAL = MeasuredResidual(
    rhs_norm=RHS_NORM, true_residual_rel=UNCONVERGED_TRUE_RESIDUAL
)


def residual_history(attempt: SolveAttempt) -> tuple[float, ...]:
    """The stored relative residuals of an attempt: geometric from 1 to its true residual.

    An attempt with a measured residual and ``k >= 1`` iterations falls by the same factor every
    iteration, so its convergence factor is ``true_residual_rel ** (1 / k)``; one without either
    stored no history.
    """
    iterations = attempt.outcome.iterations
    match attempt.residual:
        case MeasuredResidual(true_residual_rel=final) if iterations >= 1:
            return tuple(final ** (step / iterations) for step in range(iterations + 1))
        case MeasuredResidual() | UnmeasuredResidual():
            return ()


def convergence_frame(attempts: Collection[SolveAttempt]) -> pl.DataFrame:
    """The ``convergence`` frame of ``attempts``, each with its ``residual_history``."""
    return pl.DataFrame(
        [
            {
                "method": str(attempt.method),
                "system_id": str(attempt.system_id),
                "iteration": step,
                "residual_rel": residual,
                "energy_error_rel": None,
            }
            for attempt in attempts
            for step, residual in enumerate(residual_history(attempt))
        ],
        schema=CONVERGENCE_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class MethodEvents:
    """The setup events and solve attempts one method produced for ``PLANNED``."""

    method: PreconditionerName
    setup_events: tuple[MatrixSetupEvent, ...]
    solve_attempts: tuple[SolveAttempt, ...]

    def frames(self) -> MethodFrames:
        """The stored frames these events serialize to."""
        return MethodFrames(
            planned_systems=build_planned_systems_frame(provenance_for(PLANNED)),
            setup_events=build_setup_frame(self.setup_events),
            solve_attempts=build_solve_frame(self.solve_attempts),
            convergence=convergence_frame(self.solve_attempts),
        )


def _setup(
    method: PreconditionerName, matrix: int, setup_s: float, outcome: SetupOutcome
) -> MatrixSetupEvent:
    return MatrixSetupEvent(
        event_id=SetupEventId(f"{method}-setup-{matrix}"),
        dataset=DATASET,
        matrix_id=MatrixId(matrix),
        method=method,
        setup_s=setup_s,
        samples_s=(setup_s,),
        peak_memory=MeasuredPeakMemory(bytes=100),
        outcome=outcome,
    )


def _attempt(
    method: PreconditionerName,
    system: str,
    solve_s: float,
    outcome: SolveOutcome,
    residual: ResidualMeasurement,
    samples_s: tuple[float, ...] | None = None,
) -> SolveAttempt:
    tag = next(tag for tag in PLANNED if tag.system_id == system)
    return SolveAttempt(
        setup_event_id=SetupEventId(f"{method}-setup-{tag.matrix_id}"),
        system_id=tag.system_id,
        dataset=tag.dataset,
        recipe=tag.recipe,
        matrix_id=tag.matrix_id,
        rhs_ordinal=tag.rhs_ordinal,
        method=method,
        solve_s=solve_s,
        samples_s=samples_s or (solve_s,),
        peak_memory=MeasuredPeakMemory(bytes=10),
        outcome=outcome,
        residual=residual,
        exact_reference=NO_REFERENCE,
        repeats=SingleRun(),
    )


def _converged(
    method: PreconditionerName,
    system: str,
    iterations: int,
    solve_s: float,
    samples_s: tuple[float, ...] | None = None,
) -> SolveAttempt:
    return _attempt(
        method,
        system,
        solve_s,
        SolveConverged(iterations, 1e-9),
        _CONVERGED_RESIDUAL,
        samples_s,
    )


def amg_events() -> MethodEvents:
    """Four converged systems, one unconverged, one solver error, a failed setup, a gap."""
    return MethodEvents(
        method=AMG,
        setup_events=(
            _setup(AMG, 1, 6.0, SetupCompleted()),
            _setup(AMG, 2, 12.0, SetupCompleted()),
            _setup(AMG, 3, 0.5, SetupFailed("assembly failed")),
        ),
        solve_attempts=(
            _converged(AMG, "m1-0", 2, 1.0, samples_s=(0.9, 1.0, 1.2)),
            _converged(AMG, "m1-1", 5, 2.0),
            _converged(AMG, "m1-2", 3, 3.0),
            _converged(AMG, "m2-0", 8, 4.0),
            _attempt(AMG, "m2-1", 8.0, SolveFailed(20, "iteration limit"), _UNCONVERGED_RESIDUAL),
            _attempt(AMG, "m2-2", 0.5, SolveFailed(0, "CG solver failed"), NO_RESIDUAL),
        ),
    )


def jacobi_events() -> MethodEvents:
    """Every planned system converged."""
    iterations_and_seconds = {
        "m1-0": (10, 0.5),
        "m1-1": (12, 0.6),
        "m1-2": (14, 0.7),
        "m2-0": (16, 0.8),
        "m2-1": (18, 0.9),
        "m2-2": (20, 1.0),
        "m3-0": (4, 0.1),
        "m3-1": (6, 0.2),
        "m4-0": (8, 0.3),
    }
    return MethodEvents(
        method=JACOBI,
        setup_events=tuple(
            _setup(JACOBI, matrix, setup_s, SetupCompleted())
            for matrix, setup_s in ((1, 1.0), (2, 2.0), (3, 0.1), (4, 0.2))
        ),
        solve_attempts=tuple(
            _converged(JACOBI, system, iterations, solve_s)
            for system, (iterations, solve_s) in iterations_and_seconds.items()
        ),
    )


def broken_events() -> MethodEvents:
    """Setup failed on every reached matrix, so no system was solved."""
    return MethodEvents(
        method=BROKEN,
        setup_events=tuple(
            _setup(BROKEN, matrix, 0.1, SetupFailed("out of memory")) for matrix in (1, 2, 3)
        ),
        solve_attempts=(),
    )


def unbalanced_events() -> Mapping[PreconditionerName, MethodEvents]:
    """The events of all three methods, keyed by method."""
    return {events.method: events for events in (amg_events(), jacobi_events(), broken_events())}


@dataclass(frozen=True, slots=True)
class SyntheticRow:
    """One hand-written observation: a method's outcome on one system.

    ``seconds`` is the solve time. The online and standalone time of a solved system equal it
    unless ``online_seconds`` is given, which stands for solve time plus a setup share.
    """

    method: PreconditionerName
    system_id: str
    matrix_id: int
    outcome: Outcome
    seconds: float | None = None
    iterations: int | None = None
    dataset: DatasetId = DATASET
    online_seconds: float | None = None


def solved_row(
    method: PreconditionerName,
    system_id: str,
    matrix_id: int,
    seconds: float,
    iterations: int = 1,
    dataset: DatasetId = DATASET,
    online_seconds: float | None = None,
) -> SyntheticRow:
    """A converged system whose solve took ``seconds``."""
    return SyntheticRow(
        method,
        system_id,
        matrix_id,
        Outcome.CONVERGED,
        seconds,
        iterations,
        dataset,
        online_seconds,
    )


def unconverged_row(
    method: PreconditionerName, system_id: str, matrix_id: int, dataset: DatasetId = DATASET
) -> SyntheticRow:
    """A system whose solve returned an iterate that missed the tolerance."""
    return SyntheticRow(method, system_id, matrix_id, Outcome.UNCONVERGED, dataset=dataset)


def synthetic_observations(rows: Sequence[SyntheticRow]) -> pl.DataFrame:
    """Observation frame of ``rows`` for tests that need exact hand-chosen values.

    Columns a row does not state are null, except the identity columns and the planned RHS
    count, which are the same for every row.
    """
    return pl.DataFrame(
        [
            {
                "dataset": row.dataset,
                "recipe": "synthetic",
                "matrix_id": row.matrix_id,
                "system_id": row.system_id,
                "rhs_ordinal": 0,
                "method": row.method,
                "outcome": row.outcome.value,
                "iterations": row.iterations,
                "solve_s": row.seconds,
                "setup_s": None,
                "planned_rhs": 1,
                "online_s": row.seconds if row.online_seconds is None else row.online_seconds,
                "standalone_s": row.seconds,
            }
            for row in rows
        ],
        schema=OBSERVATION_SCHEMA,
    )


SECOND_RECIPE_MATRICES = (3, 4)
"""The matrices ``with_second_recipe`` moves to the second recipe."""


def with_second_recipe(observations: pl.DataFrame) -> pl.DataFrame:
    """``observations`` with the systems of ``SECOND_RECIPE_MATRICES`` drawn from ``SINE``.

    The unbalanced workload then mixes two recipes: matrices 1 and 2 stay Gaussian, so every
    outcome but setup failure stays in the first recipe.
    """
    return observations.with_columns(
        recipe=pl.when(pl.col("matrix_id").is_in(SECOND_RECIPE_MATRICES))
        .then(pl.lit(recipe_key(SINE)))
        .otherwise(pl.col("recipe"))
    )


def observations_of_events(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    solve_attempts: Collection[SolveAttempt],
    *,
    planned_by_method: Mapping[PreconditionerName, Sequence[SystemTag]] | None = None,
) -> pl.DataFrame:
    """The observation view of hand-built events, as a report is given it.

    Each method is planned for ``planned_by_method[method]``; without it, every method with a
    setup event is planned for every tag of ``planned``. All systems run under
    ``provenance_for``'s solver context.
    """
    methods = (
        dict(planned_by_method)
        if planned_by_method is not None
        else dict.fromkeys((event.method for event in setup_events), planned)
    )
    return observations_of_methods(
        {
            method: MethodFrames(
                planned_systems=build_planned_systems_frame(provenance_for(tuple(tags))),
                setup_events=build_setup_frame(
                    [event for event in setup_events if event.method == method]
                ),
                solve_attempts=build_solve_frame(
                    [attempt for attempt in solve_attempts if attempt.method == method]
                ),
                convergence=convergence_frame(
                    [attempt for attempt in solve_attempts if attempt.method == method]
                ),
            )
            for method, tags in methods.items()
        }
    )


def observations_of_workloads(workloads: Sequence[CompletedMethodWorkload]) -> pl.DataFrame:
    """The observation view of completed workloads, each method planned for its own ``planned``."""
    return observations_of_events(
        [tag for workload in workloads for tag in workload.planned],
        [event for workload in workloads for event in workload.setup_events],
        [attempt for workload in workloads for attempt in workload.solve_attempts],
        planned_by_method={workload.method: workload.planned for workload in workloads},
    )
