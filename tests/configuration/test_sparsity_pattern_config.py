"""The ``[output].sparsity_pattern`` key: default, accepted values, and rejection of others."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.data_models import OutputConfig
from neuralls.shared.types import SparsityPattern

_UNKNOWN_PATTERN = "banded"


def test_default_sparsity_pattern_is_ragged() -> None:
    assert OutputConfig().sparsity_pattern is SparsityPattern.RAGGED


@pytest.mark.parametrize("value", ["shared", "ragged"])
def test_allowed_sparsity_pattern_values_load(value: str) -> None:
    config = OutputConfig.model_validate({"sparsity_pattern": value})
    assert config.sparsity_pattern == SparsityPattern(value)


def test_unknown_sparsity_pattern_is_rejected_naming_allowed_values() -> None:
    with pytest.raises(ValidationError, match="shared") as excinfo:
        OutputConfig.model_validate({"sparsity_pattern": _UNKNOWN_PATTERN})
    assert "ragged" in str(excinfo.value)


def test_sparsity_pattern_is_part_of_the_digest() -> None:
    """A value that changes the stored layout changes the dataset, so it is not Cosmetic."""
    from neuralls.shared.digest import Cosmetic

    field = OutputConfig.model_fields["sparsity_pattern"]
    assert not any(isinstance(item, Cosmetic) for item in field.metadata)
