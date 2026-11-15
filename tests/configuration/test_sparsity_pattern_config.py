"""The ``[output].sparsity_pattern`` key: default and accepted values."""

from __future__ import annotations

import pytest

from neuralls.platform.config.models.data_models import OutputConfig
from neuralls.shared.digest import Cosmetic
from neuralls.shared.types import SparsityPattern


@pytest.mark.parametrize("value", ["shared", "ragged"])
def test_allowed_sparsity_pattern_values_load(value: str) -> None:
    config = OutputConfig.model_validate({"sparsity_pattern": value})
    assert config.sparsity_pattern == SparsityPattern(value)


def test_sparsity_pattern_is_part_of_the_digest() -> None:
    """A value that changes the stored layout changes the dataset, so it is not Cosmetic."""
    field = OutputConfig.model_fields["sparsity_pattern"]
    assert not any(isinstance(item, Cosmetic) for item in field.metadata)
