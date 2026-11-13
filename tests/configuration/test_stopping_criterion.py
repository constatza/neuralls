"""The solver has one stopping rule, so any other configured value is rejected at load."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.comparison import parse_comparison_method_override
from neuralls.platform.config.models.experiments import ComparisonBaseline


def test_baseline_rejects_fixed_iterations() -> None:
    with pytest.raises(ValidationError):
        ComparisonBaseline(stopping_criterion="fixed_iterations")


def test_method_override_rejects_fixed_iterations() -> None:
    context = ConfigContext(config_path=Path("override.toml"), settings=None, home_dir=Path("home"))

    with pytest.raises(ValidationError):
        parse_comparison_method_override(
            {"general": {"params": {"stopping_criterion": "fixed_iterations"}}}, context
        )
