"""Tests for ``FigureStyleConfig`` and how its rotation reaches a figure's style."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.platform.config.models.experiments import ComparisonBaseline
from neuralls.platform.config.models.figure_style import FigureStyleConfig
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.group_plots import GroupStyler
from neuralls.shared.constants import DEFAULT_LABEL_ROTATION_DEG, MAX_LABEL_ROTATION_DEG
from neuralls.shared.scientific_description.policy import NotationPolicy


def test_default_needs_no_section() -> None:
    assert FigureStyleConfig().label_rotation_deg == DEFAULT_LABEL_ROTATION_DEG


@pytest.mark.parametrize("degrees", [0.0, 45.0, MAX_LABEL_ROTATION_DEG])
def test_rotation_inside_bounds_is_kept(degrees: float) -> None:
    assert (
        FigureStyleConfig.model_validate({"label_rotation_deg": degrees}).label_rotation_deg
        == degrees
    )


@pytest.mark.parametrize("degrees", [-0.1, MAX_LABEL_ROTATION_DEG + 0.1])
def test_rotation_outside_bounds_is_rejected(degrees: float) -> None:
    with pytest.raises(ValidationError):
        FigureStyleConfig.model_validate({"label_rotation_deg": degrees})


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FigureStyleConfig.model_validate({"rotation": 10})


def test_baseline_section_is_optional_and_parsed_from_a_dict() -> None:
    assert ComparisonBaseline.model_validate({}).figure_style == FigureStyleConfig()
    parsed = ComparisonBaseline.model_validate({"figure_style": {"label_rotation_deg": 60.0}})
    assert parsed.figure_style.label_rotation_deg == 60.0


def test_styler_carries_the_configured_rotation_into_every_figure_style(
    profile_report: RecipeProfileReport,
    repeated_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
) -> None:
    styler = GroupStyler(
        repeated_timing_presentation, notation, FigureStyleConfig(label_rotation_deg=55.0)
    )
    plans = figure_plans(profile_report, repeated_timing_presentation, "key", {})

    assert {styler.style(plan).label_rotation_deg for plan in plans} == {55.0}
    assert plans
