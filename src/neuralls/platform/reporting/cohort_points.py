"""The observation rows behind the panel of one cohort of a report figure."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population, select


class CohortPanel(Protocol):
    """A panel of one cohort, drawn over the systems of one population."""

    @property
    def cohort(self) -> str: ...

    @property
    def population(self) -> Population: ...


def cohort_systems(
    observations: pl.DataFrame, panel: CohortPanel, metric: MetricId
) -> pl.DataFrame:
    """The rows of ``observations`` that define ``metric`` for the population of ``panel``.

    ``observations`` are those of the panel's cohort already (``cohort_observations``), so the
    recipe is not filtered again here.
    """
    return select(observations, metric, panel.population)


def panel_points(observations: pl.DataFrame, panel: CohortPanel, metric: MetricId) -> pl.DataFrame:
    """The per-system values behind ``panel``: its recipe's selected systems, ``method, value``."""
    return cohort_systems(observations, panel, metric).select(
        "method", value=METRICS[metric].column
    )
