"""The observation rows behind one recipe-cohort panel of a report figure."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from neuralls.domain.solver.reporting.metrics import METRICS, MetricId
from neuralls.domain.solver.reporting.summary import Population, select


class CohortPanel(Protocol):
    """A panel of one recipe cohort, drawn over the systems of one population."""

    @property
    def cohort(self) -> str: ...

    @property
    def population(self) -> Population: ...


def cohort_systems(
    observations: pl.DataFrame, panel: CohortPanel, metric: MetricId
) -> pl.DataFrame:
    """The observation rows behind ``panel``: its recipe's systems that define ``metric``."""
    return select(observations, metric, panel.population).filter(pl.col("recipe") == panel.cohort)


def panel_points(observations: pl.DataFrame, panel: CohortPanel, metric: MetricId) -> pl.DataFrame:
    """The per-system values behind ``panel``: its recipe's selected systems, ``method, value``."""
    return cohort_systems(observations, panel, metric).select(
        "method", value=METRICS[metric].column
    )
