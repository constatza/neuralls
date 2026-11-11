"""``show_titles`` defaults on and reads from the ``[figure_style]`` table."""

from __future__ import annotations

from neuralls.platform.config.models.figure_style import FigureStyleConfig


def test_titles_are_shown_by_default() -> None:
    assert FigureStyleConfig().show_titles is True


def test_titles_can_be_turned_off() -> None:
    assert FigureStyleConfig.model_validate({"show_titles": False}).show_titles is False
