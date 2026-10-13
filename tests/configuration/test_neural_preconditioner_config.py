"""Tests for NeuralPreconditionerConfig extra_input_names field."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.preconditioner import NeuralPreconditionerConfig


def test_extra_input_names_defaults_empty() -> None:
    """extra_input_names defaults to empty tuple when not provided."""
    cfg = NeuralPreconditionerConfig()
    assert cfg.extra_input_names == ()


def test_extra_input_names_from_list() -> None:
    """extra_input_names accepts a list and converts it to tuple."""
    cfg = NeuralPreconditionerConfig(extra_input_names=["matrix", "coordinates"])
    assert cfg.extra_input_names == ("matrix", "coordinates")


def test_extra_input_names_from_tuple() -> None:
    """extra_input_names accepts a tuple directly."""
    cfg = NeuralPreconditionerConfig(extra_input_names=("matrix",))
    assert cfg.extra_input_names == ("matrix",)


def test_config_is_frozen() -> None:
    """NeuralPreconditionerConfig is immutable (frozen)."""
    cfg = NeuralPreconditionerConfig(extra_input_names=("matrix",))
    with pytest.raises((ValidationError, AttributeError)):
        cfg.extra_input_names = ("other",)  # type: ignore[misc]
