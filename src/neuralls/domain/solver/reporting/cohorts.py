"""Per-method statistics of a class-level report, labelled by the RHS recipe they summarize.

Raw-scale statistics (means, quantiles, ratios of sums) are drawn only at the
matrix-set-and-RHS-type level, so a table holds the block of its one recipe; the ``cohort``
column names it with the recipe's lossless key (``shared/rhs_recipe.py``). Every statistic keeps
its own schema behind the ``cohort`` column. ``Cohort`` says what a figure draws: the whole
class-level group, or every system of a mixture level pooled with equal weight per pair.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from neuralls.domain.solver.reporting.summary import RATIO_OF_SUMS_SCHEMA, SUMMARY_SCHEMA

type Schema = Mapping[str, pl.DataType | type[pl.DataType]]

COHORT_COLUMN = "cohort"


@dataclass(frozen=True, slots=True)
class WholeGroup:
    """The figure draws the whole class-level group: one matrix set and one RHS recipe."""


@dataclass(frozen=True, slots=True)
class MixtureCohort:
    """The figure draws every system of a mixture level pooled, each pair weighted equally."""


type Cohort = WholeGroup | MixtureCohort
"""What a figure draws: one class, or a mixture of classes or matrix sets."""


def pools_a_mixture(n_recipes: int, n_matrix_sets: int) -> bool:
    """Whether systems of ``n_recipes`` recipes over ``n_matrix_sets`` matrix sets are a mixture.

    One recipe over one matrix set is a class: its systems are exactly a class-level group's.
    """
    return n_recipes >= 2 or n_matrix_sets >= 2


def mixture_label(recipe_names: Sequence[str], n_matrix_sets: int) -> str:
    """How a figure pooled over a mixture names what it pools; each pair counts once.

    Several recipes are an equal-weight mixture of them; one recipe over several matrix sets is
    pooled over those sets; several of both says both.

    Raises:
        ValueError: One recipe over one matrix set, which is a class, not a mixture.
    """
    if not pools_a_mixture(len(recipe_names), n_matrix_sets):
        raise ValueError("one recipe over one matrix set is a class, not a mixture")
    parts = (
        f"equal-weight mixture of {', '.join(recipe_names)}" if len(recipe_names) >= 2 else None,
        f"pooled over {n_matrix_sets} matrix sets" if n_matrix_sets >= 2 else None,
    )
    return "; ".join(part for part in parts if part is not None)


class NotOneCohortError(ValueError):
    """A figure was given a table holding no cohort, or several, where it draws exactly one."""


def cohort_schema(schema: Schema) -> dict[str, pl.DataType | type[pl.DataType]]:
    """``schema`` behind a leading ``cohort`` column."""
    return {COHORT_COLUMN: pl.String, **schema}


COHORT_SUMMARY_SCHEMA = cohort_schema(SUMMARY_SCHEMA)
COHORT_RATIO_OF_SUMS_SCHEMA = cohort_schema(RATIO_OF_SUMS_SCHEMA)


def by_cohort(
    obs: pl.DataFrame,
    statistic: Callable[[pl.DataFrame], pl.DataFrame],
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
    return pl.concat([pl.DataFrame(schema=cohort_schema(schema)), *blocks], how="vertical")


def _labelled(frame: pl.DataFrame, cohort: str) -> pl.DataFrame:
    return frame.select(pl.lit(cohort).alias(COHORT_COLUMN), pl.all())


def single_cohort(table: pl.DataFrame) -> str:
    """The one cohort ``table`` holds, as its ``cohort`` column names it."""
    names = table[COHORT_COLUMN].unique().to_list()
    if len(names) != 1:
        raise NotOneCohortError(f"expected exactly one cohort, found {len(names)}")
    return str(names[0])
