"""Identity of a planned system in the report tables: its RHS recipe and tag.

A recipe is the *kind and parameters* of how a right-hand side is made, so two
sinusoidal RHS with different omegas are different recipes while entries with
identical parameters share one. A raw vector source is identified by its row
and a fingerprint of its path, row kind and scale, so two files never collapse
into one recipe.
"""

from __future__ import annotations

from typing import assert_never

from neuralls.composition.comparison.models import PlannedSystem
from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.platform.config.models.comparison import (
    ComparisonRhsSourceModel,
    DatasetRhsSourceModel,
    GaussianRhsSourceModel,
    NoMask,
    RandomMask,
    RawLhsSourceModel,
    RawRhsSourceModel,
    RawVectorSourceBase,
    RhsMask,
    SinusoidalRhsSourceModel,
)
from neuralls.shared.digest import canonical_digest
from neuralls.shared.rhs_recipe import (
    DatasetRecipe,
    GaussianRecipe,
    RandomlyMasked,
    RawLhsRecipe,
    RawRhsRecipe,
    RawSourceIdentity,
    RecipeMask,
    RhsRecipe,
    SinusoidalRecipe,
    Unmasked,
)
from neuralls.shared.types import DatasetId, SystemId

__all__ = ["rhs_recipe", "system_tag"]

_FINGERPRINT_HEX_LENGTH = 16


def rhs_recipe(source: ComparisonRhsSourceModel) -> RhsRecipe:
    """The recipe of a configured RHS source."""
    mask = _recipe_mask(source.mask)
    match source:
        case GaussianRhsSourceModel(mean=mean, std=std):
            return GaussianRecipe(mean, std, mask)
        case SinusoidalRhsSourceModel(omega=omega):
            return SinusoidalRecipe(omega, mask)
        case RawLhsSourceModel():
            return RawLhsRecipe(_raw_identity(source), mask)
        case RawRhsSourceModel():
            return RawRhsRecipe(_raw_identity(source), mask)
        case DatasetRhsSourceModel():
            return DatasetRecipe(mask)
        case _:
            assert_never(source)


def _recipe_mask(mask: RhsMask) -> RecipeMask:
    match mask:
        case NoMask():
            return Unmasked()
        case RandomMask(density=density):
            return RandomlyMasked(density)
        case _:
            assert_never(mask)


def _raw_identity(source: RawVectorSourceBase) -> RawSourceIdentity:
    row_kind = None if source.row_kind is None else source.row_kind.name
    digest = canonical_digest(source.path.as_posix(), source.sample_index, row_kind, source.scale)
    fingerprint = digest.rpartition(":")[2][:_FINGERPRINT_HEX_LENGTH]
    return RawSourceIdentity(sample_index=source.sample_index, fingerprint=fingerprint)


def system_tag(planned: PlannedSystem) -> SystemTag:
    """Where a planned system sits in the report tables."""
    return SystemTag(
        dataset=DatasetId(planned.entry.matrix_dataset),
        recipe=rhs_recipe(planned.entry.rhs_source),
        matrix_id=planned.matrix.matrix_id,
        rhs_ordinal=planned.rhs_ordinal,
        system_id=SystemId(planned.entry.id),
    )
