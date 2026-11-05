"""The observation view of a stored session, rebuilt from its method artifacts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.composition.comparison.postprocessing.report_inputs import observation_view
from neuralls.domain.presentation.methods import ExecutedMethod, PreparedComparisonPresentation
from neuralls.domain.solver.models.cost import NO_OFFLINE_COST
from neuralls.domain.solver.reporting.observations import observations_of_methods
from neuralls.platform.config.models.preconditioner import parse_preconditioner_config
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.types import PreconditionerName
from tests.observation_support import (
    AMG,
    JACOBI,
    PLANNED,
    MethodEvents,
    amg_events,
    jacobi_events,
)
from tests.provenance_support import provenance_for


@pytest.fixture
def stored_events() -> dict[PreconditionerName, MethodEvents]:
    """The events ``amg`` and ``jacobi`` stored."""
    return {AMG: amg_events(), JACOBI: jacobi_events()}


@pytest.fixture
def stored_methods(
    prepared_cohort: Callable[..., PreparedComparisonPresentation],
    stored_events: dict[PreconditionerName, MethodEvents],
) -> tuple[StoredMethod, ...]:
    """``amg`` and ``jacobi`` as stored: validated provenance plus the event and convergence tables."""
    cohort = prepared_cohort(str(AMG), str(JACOBI))
    return tuple(
        StoredMethod(
            run_id=f"run-{method}",
            name=method,
            config_digest="sha256:" + "d" * 64,
            config=parse_preconditioner_config({"name": str(method), "type": "identity"}),
            presentation=ExecutedMethod(cohort.method(method), ()),
            offline=NO_OFFLINE_COST,
            systems=provenance_for(PLANNED),
            setup_events=events.frames().setup_events,
            solve_attempts=events.frames().solve_attempts,
            convergence=events.frames().convergence,
        )
        for method, events in stored_events.items()
    )


def test_the_view_of_a_stored_session_is_the_view_of_its_event_frames(
    stored_methods: tuple[StoredMethod, ...],
    stored_events: dict[PreconditionerName, MethodEvents],
) -> None:
    view = observation_view(stored_methods)

    expected = observations_of_methods(
        {method: events.frames() for method, events in stored_events.items()}
    )
    assert view.equals(expected)
    assert view.height == 18
