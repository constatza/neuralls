"""Per-method statistics split by RHS recipe, with the pooled rows of a mixture beside them.

A report level that pools several recipes is a mixture: its pooled numbers hide how the recipes
differ, so a figure draws one panel per recipe and the pooled rows appear only in its table,
labelled ``MIXTURE_COHORT``. A *cohort* is the block of rows of one recipe, or that pooled block;
the ``cohort`` column names it with the recipe's lossless key (``shared/rhs_recipe.py``), which
never equals ``MIXTURE_COHORT``. Every statistic keeps its own schema behind the ``cohort``
column, so one table holds exactly what each panel draws and what the mixture pools.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.solver.reporting.summary import RATIO_OF_SUMS_SCHEMA, SUMMARY_SCHEMA

type Schema = Mapping[str, pl.DataType | type[pl.DataType]]

COHORT_COLUMN = "cohort"
MIXTURE_COHORT = "mixture"
"""The cohort of the rows pooled over every recipe of a mixture level."""


class CohortRows(StrEnum):
    """Which cohorts a table holds, decided by whether its report level pools recipes."""

    RECIPES = "recipes"
    """One block per recipe; a single-recipe level, whose pooled rows would repeat its block."""
    RECIPES_AND_MIXTURE = "recipes_and_mixture"
    """One block per recipe, then the block pooled over all of them."""


def cohort_schema(schema: Schema) -> dict[str, pl.DataType | type[pl.DataType]]:
    """``schema`` behind a leading ``cohort`` column."""
    return {COHORT_COLUMN: pl.String, **schema}


COHORT_SUMMARY_SCHEMA = cohort_schema(SUMMARY_SCHEMA)
COHORT_RATIO_OF_SUMS_SCHEMA = cohort_schema(RATIO_OF_SUMS_SCHEMA)


def by_cohort(
    obs: pl.DataFrame,
    statistic: Callable[[pl.DataFrame], pl.DataFrame],
    rows: CohortRows,
    *,
    schema: Schema,
) -> pl.DataFrame:
    """``statistic`` of each recipe's rows of ``obs``, in order of first appearance, by cohort.

    ``schema`` is the schema of the frames ``statistic`` returns; the result has
    ``cohort_schema(schema)``. Each recipe's statistic sees only that recipe's systems, so a
    common population is taken within the recipe.
    """
    recipes = obs["recipe"].unique(maintain_order=True).to_list()
    blocks = [_labelled(statistic(obs.filter(pl.col("recipe") == key)), key) for key in recipes]
    match rows:
        case CohortRows.RECIPES:
            pass
        case CohortRows.RECIPES_AND_MIXTURE:
            blocks.append(_labelled(statistic(obs), MIXTURE_COHORT))
        case unreachable:
            assert_never(unreachable)
    return pl.concat([pl.DataFrame(schema=cohort_schema(schema)), *blocks], how="vertical")


def _labelled(frame: pl.DataFrame, cohort: str) -> pl.DataFrame:
    return frame.select(pl.lit(cohort).alias(COHORT_COLUMN), pl.all())
