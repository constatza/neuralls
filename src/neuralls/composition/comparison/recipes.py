"""Identity of a planned system in the report tables: its RHS recipe and tag.

A recipe is the *kind and parameters* of how a right-hand side is made, so two
sinusoidal RHS with different omegas are different recipes while entries with
identical parameters share one. The label is each source model's
`recipe_label`, so every report level reads the one definition.
"""

from __future__ import annotations

from neuralls.composition.comparison.models import PlannedSystem
from neuralls.domain.solver.metrics_table import SystemTag
from neuralls.shared.types import DatasetId

__all__ = ["system_tag"]


def system_tag(planned: PlannedSystem) -> SystemTag:
    """Where a planned system sits in the report tables."""
    return SystemTag(
        dataset=DatasetId(planned.entry.matrix_dataset),
        recipe=planned.entry.rhs_source.recipe_label,
        matrix_id=planned.matrix.matrix_id,
        rhs_ordinal=planned.rhs_ordinal,
        system_id=planned.entry.id,
    )
