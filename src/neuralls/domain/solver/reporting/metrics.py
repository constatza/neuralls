r"""The catalog of per-system and per-matrix metrics a report can summarize.

Each ``MetricSpec`` is the single source for the observation column a metric reads, its unit and
label, its LaTeX formula, and the outcomes on which it is defined. The last is the failure rule:
a system whose outcome is outside ``defined_on`` has no value for the metric and is excluded from
every mean, quantile and interval of it, never counted as zero and never given the iteration cap.
``domain/solver/README.md`` carries the same labels and formulas verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from neuralls.domain.solver.reporting.observations import Outcome


class MetricId(StrEnum):
    """The closed set of summarized metrics."""

    ITERATIONS = "iterations"
    SOLVE_S = "solve_s"
    ONLINE_S = "online_s"
    STANDALONE_S = "standalone_s"
    SETUP_S = "setup_s"
    TIME_PER_ITERATION_S = "time_per_iteration_s"


class Unit(StrEnum):
    """What a metric value is counted in."""

    ITERATIONS = "iterations"
    SECONDS = "s"
    SECONDS_PER_ITERATION = "s/iteration"


class ObservationUnit(StrEnum):
    """What one value of a metric belongs to."""

    SYSTEM = "system"
    MATRIX = "matrix"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One metric: where it is read, how it is labelled, and on which outcomes it exists."""

    id: MetricId
    column: str
    unit: Unit
    label: str
    observation_unit: ObservationUnit
    defined_on: frozenset[Outcome]
    formula: str


_CONVERGED_ONLY = frozenset({Outcome.CONVERGED})
_RETURNED_AN_ITERATE = frozenset({Outcome.CONVERGED, Outcome.UNCONVERGED})
_SETUP_COMPLETED = frozenset({Outcome.CONVERGED, Outcome.UNCONVERGED, Outcome.SOLVER_ERROR})

_SPECS = (
    MetricSpec(
        id=MetricId.ITERATIONS,
        column="iterations",
        unit=Unit.ITERATIONS,
        label="iterations to converge",
        observation_unit=ObservationUnit.SYSTEM,
        defined_on=_CONVERGED_ONLY,
        formula=r"k_{p,s},\quad s \in \mathcal{C}_p",
    ),
    MetricSpec(
        id=MetricId.SOLVE_S,
        column="solve_s",
        unit=Unit.SECONDS,
        label="solve time per system [s]",
        observation_unit=ObservationUnit.SYSTEM,
        defined_on=_CONVERGED_ONLY,
        formula=r"t^{\mathrm{solve}}_{p,s} = \operatorname{stat}_{j}\,\tau^{\mathrm{solve}}_{p,s,j}",
    ),
    MetricSpec(
        id=MetricId.ONLINE_S,
        column="online_s",
        unit=Unit.SECONDS,
        label="online time per system [s]",
        observation_unit=ObservationUnit.SYSTEM,
        defined_on=_CONVERGED_ONLY,
        formula=(
            r"t^{\mathrm{online}}_{p,s} = t^{\mathrm{solve}}_{p,s}"
            r" + \frac{t^{\mathrm{setup}}_{p,m(s)}}{r_{m(s)}}"
        ),
    ),
    MetricSpec(
        id=MetricId.STANDALONE_S,
        column="standalone_s",
        unit=Unit.SECONDS,
        label="standalone time per system [s]",
        observation_unit=ObservationUnit.SYSTEM,
        defined_on=_CONVERGED_ONLY,
        formula=(
            r"t^{\mathrm{standalone}}_{p,s} = t^{\mathrm{solve}}_{p,s}"
            r" + t^{\mathrm{setup}}_{p,m(s)}"
        ),
    ),
    MetricSpec(
        id=MetricId.SETUP_S,
        column="setup_s",
        unit=Unit.SECONDS,
        label="setup time per matrix [s]",
        observation_unit=ObservationUnit.MATRIX,
        defined_on=_SETUP_COMPLETED,
        formula=r"t^{\mathrm{setup}}_{p,m} = \operatorname{stat}_{j}\,\tau^{\mathrm{setup}}_{p,m,j}",
    ),
    MetricSpec(
        id=MetricId.TIME_PER_ITERATION_S,
        column="time_per_iteration_s",
        unit=Unit.SECONDS_PER_ITERATION,
        label="time per iteration [s]",
        observation_unit=ObservationUnit.SYSTEM,
        defined_on=_RETURNED_AN_ITERATE,
        formula=r"\theta_{p,s} = \frac{t^{\mathrm{solve}}_{p,s}}{k_{p,s}}",
    ),
)

METRICS: Mapping[MetricId, MetricSpec] = MappingProxyType({spec.id: spec for spec in _SPECS})
"""Every metric by id; read-only so the failure rules cannot be changed at run time."""


def system_metric(metric: MetricId) -> MetricSpec:
    """The spec of a metric that has one value per system, for paired and profile statistics.

    Raises:
        ValueError: The metric has one value per matrix, so it cannot be paired by system.
    """
    spec = METRICS[metric]
    if spec.observation_unit is not ObservationUnit.SYSTEM:
        raise ValueError(
            f"metric {metric} has one value per {spec.observation_unit}, not per system"
        )
    return spec
