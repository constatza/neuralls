"""Explicit comparison-reference configuration and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.domain.benchmarking.selection import (
    ReferenceUnavailable,
    resolve_reference_selection,
    validate_reference_preconditioner,
)
from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.comparison import parse_comparison_method_override
from neuralls.shared.types import PreconditionerName
from tests.workflows.comparison_support import virtual_settings


def test_method_override_reference_is_explicit_and_can_clear_baseline() -> None:
    override = parse_comparison_method_override(
        {"general": {"reference_preconditioner": None}},
        ConfigContext(
            config_path=Path("override.toml"), settings=virtual_settings(), home_dir=Path("home")
        ),
    )

    assert "reference_preconditioner" in override.general.model_fields_set
    assert override.general.reference_preconditioner is None


def test_static_reference_validation_names_available_methods() -> None:
    with pytest.raises(ValueError):
        validate_reference_preconditioner("amg", ("identity", "jacobi"))


def test_runtime_resolution_skip_does_not_fallback_to_observed_method() -> None:
    selection = resolve_reference_selection(
        "amg",
        ("jacobi",),
        skipped={"amg": "checkpoint not found"},
    )

    assert isinstance(selection, ReferenceUnavailable)
    assert selection.name == PreconditionerName("amg")
    assert selection.reason == "checkpoint not found"
