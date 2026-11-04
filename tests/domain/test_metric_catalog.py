"""The metric catalog states, for every metric, its column, unit, label and failure rule."""

from __future__ import annotations

from collections.abc import MutableMapping

import pytest

from neuralls.domain.solver.reporting.metrics import (
    METRICS,
    MetricId,
    ObservationUnit,
    Unit,
)
from neuralls.domain.solver.reporting.observations import OBSERVATION_SCHEMA, Outcome

SOLVED_ONLY = frozenset({Outcome.CONVERGED})


def test_every_metric_id_has_one_spec_that_names_it() -> None:
    assert set(METRICS) == set(MetricId)
    assert all(spec.id == metric for metric, spec in METRICS.items())


def test_every_metric_reads_a_column_of_the_observation_view() -> None:
    assert all(spec.column in OBSERVATION_SCHEMA for spec in METRICS.values())


def test_the_catalog_cannot_be_mutated() -> None:
    assert not isinstance(METRICS, MutableMapping)


@pytest.mark.parametrize(
    "metric", [MetricId.ITERATIONS, MetricId.SOLVE_S, MetricId.ONLINE_S, MetricId.STANDALONE_S]
)
def test_costs_are_defined_on_converged_systems_only(metric: MetricId) -> None:
    assert METRICS[metric].defined_on == SOLVED_ONLY
    assert METRICS[metric].observation_unit is ObservationUnit.SYSTEM


def test_time_per_iteration_also_counts_unconverged_systems() -> None:
    spec = METRICS[MetricId.TIME_PER_ITERATION_S]

    assert spec.defined_on == {Outcome.CONVERGED, Outcome.UNCONVERGED}
    assert spec.observation_unit is ObservationUnit.SYSTEM
    assert spec.unit is Unit.SECONDS_PER_ITERATION


def test_setup_time_is_one_value_per_matrix_with_a_completed_setup() -> None:
    spec = METRICS[MetricId.SETUP_S]

    assert spec.observation_unit is ObservationUnit.MATRIX
    assert spec.defined_on == {Outcome.CONVERGED, Outcome.UNCONVERGED, Outcome.SOLVER_ERROR}


def test_labels_and_formulas_are_distinct() -> None:
    labels = [spec.label for spec in METRICS.values()]
    formulas = [spec.formula for spec in METRICS.values()]

    assert len(set(labels)) == len(labels)
    assert len(set(formulas)) == len(formulas)
