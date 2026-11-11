"""``show_titles`` defaults off and reads from the ``[figure_style]`` table."""

from __future__ import annotations

from neuralls.platform.config.models.figure_style import FigureStyleConfig


def test_titles_are_hidden_by_default() -> None:
    assert FigureStyleConfig().show_titles is False


def test_titles_can_be_turned_on() -> None:
    assert FigureStyleConfig.model_validate({"show_titles": True}).show_titles is True
