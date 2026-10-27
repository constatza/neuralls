"""How a comparison right-hand side is made, as a closed sum type.

A recipe is the kind of an RHS source plus the parameters that make problems
differ. Two sinusoidal sources with different frequencies, two raw files, or a
masked and an unmasked run of one kind are different recipes, so they are
reported as separate problem families. The machine key is lossless (parsing it
returns the recipe) and uses only characters that are safe in file names and
MLflow metric keys; people never read it, `domain.presentation` renders recipes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NewType, assert_never

__all__ = [
    "DatasetRecipe",
    "GaussianRecipe",
    "RandomlyMasked",
    "RawLhsRecipe",
    "RawRhsRecipe",
    "RawSourceIdentity",
    "RecipeKey",
    "RecipeMask",
    "RhsRecipe",
    "SinusoidalRecipe",
    "Unmasked",
    "parse_recipe_key",
    "recipe_key",
]

RecipeKey = NewType("RecipeKey", str)
"""The lossless machine form of an `RhsRecipe` (file-name and metric-key safe)."""

_SEPARATOR = "_"
_FINGERPRINT = re.compile(r"[0-9a-f]+")

_GAUSSIAN = "gaussian"
_SINUSOIDAL = "sinusoidal"
_RAW_LHS = "raw-lhs"
_RAW_RHS = "raw-rhs"
_DATASET = "dataset"

_MEAN, _STD, _OMEGA, _SAMPLE, _FINGERPRINT_TAG, _DENSITY = "m", "s", "w", "i", "f", "r"


@dataclass(frozen=True, slots=True)
class Unmasked:
    """Every entry of the RHS is kept."""


@dataclass(frozen=True, slots=True)
class RandomlyMasked:
    """All but a random fraction ``density`` of the RHS entries are zeroed."""

    density: float


type RecipeMask = Unmasked | RandomlyMasked


@dataclass(frozen=True, slots=True)
class RawSourceIdentity:
    """Which raw vector a raw recipe reads: its row and a fingerprint of the whole source.

    The fingerprint covers the file path, row kind and scale, so two files, or
    two settings of one file, never share a recipe even at the same row.
    """

    sample_index: int
    fingerprint: str

    def __post_init__(self) -> None:
        """Reject values that the machine key could not represent."""
        if self.sample_index < 0:
            raise ValueError(f"sample index must be non-negative, got {self.sample_index}")
        if not _FINGERPRINT.fullmatch(self.fingerprint):
            raise ValueError(f"fingerprint must be lowercase hex, got {self.fingerprint!r}")


@dataclass(frozen=True, slots=True)
class GaussianRecipe:
    """Independent normal entries with the given mean and standard deviation."""

    mean: float
    std: float
    mask: RecipeMask = field(default_factory=Unmasked)


@dataclass(frozen=True, slots=True)
class SinusoidalRecipe:
    """A sine wave sweeping ``omega`` radians across the vector."""

    omega: float
    mask: RecipeMask = field(default_factory=Unmasked)


@dataclass(frozen=True, slots=True)
class RawLhsRecipe:
    """A stored solution-side vector, converted to a right-hand side by ``A @ x``."""

    source: RawSourceIdentity
    mask: RecipeMask = field(default_factory=Unmasked)


@dataclass(frozen=True, slots=True)
class RawRhsRecipe:
    """A stored vector that already is a right-hand side."""

    source: RawSourceIdentity
    mask: RecipeMask = field(default_factory=Unmasked)


@dataclass(frozen=True, slots=True)
class DatasetRecipe:
    """Held-out rows of the comparison entry's own dataset."""

    mask: RecipeMask = field(default_factory=Unmasked)


type RhsRecipe = GaussianRecipe | SinusoidalRecipe | RawLhsRecipe | RawRhsRecipe | DatasetRecipe


def recipe_key(recipe: RhsRecipe) -> RecipeKey:
    """The lossless machine key of ``recipe``; equal recipes have equal keys and no other do."""
    match recipe:
        case GaussianRecipe(mean=mean, std=std, mask=mask):
            parts = [_GAUSSIAN, f"{_MEAN}{mean!r}", f"{_STD}{std!r}"]
        case SinusoidalRecipe(omega=omega, mask=mask):
            parts = [_SINUSOIDAL, f"{_OMEGA}{omega!r}"]
        case RawLhsRecipe(source=source, mask=mask):
            parts = [_RAW_LHS, *_source_parts(source)]
        case RawRhsRecipe(source=source, mask=mask):
            parts = [_RAW_RHS, *_source_parts(source)]
        case DatasetRecipe(mask=mask):
            parts = [_DATASET]
        case _:
            assert_never(recipe)
    match mask:
        case Unmasked():
            pass
        case RandomlyMasked(density=density):
            parts.append(f"{_DENSITY}{density!r}")
        case _:
            assert_never(mask)
    return RecipeKey(_SEPARATOR.join(parts))


def _source_parts(source: RawSourceIdentity) -> list[str]:
    return [f"{_SAMPLE}{source.sample_index}", f"{_FINGERPRINT_TAG}{source.fingerprint}"]


def parse_recipe_key(key: str) -> RhsRecipe:
    """The recipe whose machine key is ``key``.

    Raises:
        ValueError: If ``key`` is not the key of any recipe.
    """
    kind, *tokens = key.split(_SEPARATOR)
    fields: dict[str, str] = {}
    for token in tokens:
        tag, value = token[:1], token[1:]
        if not value or tag in fields:
            raise ValueError(f"malformed recipe key: {key!r}")
        fields[tag] = value
    try:
        recipe = _build(kind, fields)
    except (KeyError, ValueError) as error:
        raise ValueError(f"malformed recipe key: {key!r}") from error
    if fields:
        raise ValueError(f"malformed recipe key: {key!r}")
    return recipe


def _build(kind: str, fields: dict[str, str]) -> RhsRecipe:
    """Consume the fields of one recipe from ``fields``; leftovers mean a malformed key."""
    density = fields.pop(_DENSITY, None)
    mask = Unmasked() if density is None else RandomlyMasked(float(density))
    if kind == _GAUSSIAN:
        return GaussianRecipe(float(fields.pop(_MEAN)), float(fields.pop(_STD)), mask)
    if kind == _SINUSOIDAL:
        return SinusoidalRecipe(float(fields.pop(_OMEGA)), mask)
    if kind == _RAW_LHS:
        return RawLhsRecipe(_pop_source(fields), mask)
    if kind == _RAW_RHS:
        return RawRhsRecipe(_pop_source(fields), mask)
    if kind == _DATASET:
        return DatasetRecipe(mask)
    raise ValueError(kind)


def _pop_source(fields: dict[str, str]) -> RawSourceIdentity:
    return RawSourceIdentity(int(fields.pop(_SAMPLE)), fields.pop(_FINGERPRINT_TAG))
