"""RHS recipes: a closed sum whose machine key is lossless and filesystem safe."""

from __future__ import annotations

import re

import pytest

from neuralls.shared.rhs_recipe import (
    DatasetRecipe,
    GaussianRecipe,
    RandomlyMasked,
    RawLhsRecipe,
    RawRhsRecipe,
    RawSourceIdentity,
    RhsRecipe,
    SinusoidalRecipe,
    Unmasked,
    parse_recipe_key,
    recipe_key,
)

KEY_ALPHABET = re.compile(r"[A-Za-z0-9._-]+")


@pytest.fixture
def raw_source() -> RawSourceIdentity:
    return RawSourceIdentity(sample_index=3, fingerprint="0123456789abcdef")


@pytest.fixture
def recipes(raw_source: RawSourceIdentity) -> tuple[RhsRecipe, ...]:
    return (
        GaussianRecipe(mean=0.0, std=1.0),
        GaussianRecipe(mean=-1.5, std=2e-05, mask=RandomlyMasked(density=0.1)),
        SinusoidalRecipe(omega=6.283185307179586),
        SinusoidalRecipe(omega=2.0, mask=RandomlyMasked(density=0.25)),
        RawLhsRecipe(source=raw_source),
        RawRhsRecipe(source=raw_source, mask=RandomlyMasked(density=0.5)),
        DatasetRecipe(),
        DatasetRecipe(mask=RandomlyMasked(density=0.1)),
    )


def test_every_recipe_key_parses_back_to_the_same_recipe(recipes: tuple[RhsRecipe, ...]) -> None:
    assert [parse_recipe_key(recipe_key(recipe)) for recipe in recipes] == list(recipes)


def test_keys_are_distinct_and_filesystem_safe(recipes: tuple[RhsRecipe, ...]) -> None:
    keys = [recipe_key(recipe) for recipe in recipes]
    assert len(set(keys)) == len(keys)
    assert all(KEY_ALPHABET.fullmatch(key) for key in keys)


def test_raw_sources_with_different_identity_are_different_recipes(
    raw_source: RawSourceIdentity,
) -> None:
    other_path = RawSourceIdentity(sample_index=3, fingerprint="fedcba9876543210")
    other_sample = RawSourceIdentity(sample_index=4, fingerprint="0123456789abcdef")

    keys = {
        recipe_key(RawLhsRecipe(source=source)) for source in (raw_source, other_path, other_sample)
    }

    assert len(keys) == 3
    assert recipe_key(RawLhsRecipe(source=raw_source)) != recipe_key(
        RawRhsRecipe(source=raw_source)
    )


def test_a_mask_separates_otherwise_equal_recipes() -> None:
    assert recipe_key(GaussianRecipe(0.0, 1.0)) != recipe_key(
        GaussianRecipe(0.0, 1.0, mask=RandomlyMasked(0.1))
    )
    assert GaussianRecipe(0.0, 1.0).mask == Unmasked()


@pytest.mark.parametrize(
    "key", ["", "unknown", "gaussian_m0.0", "dataset_x1", "sinusoidal_w2.0_w3.0"]
)
def test_malformed_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        parse_recipe_key(key)
