"""Every registered cost has a bar-chart style, and no style is orphaned."""

from __future__ import annotations

from neuralls.composition.comparison._plots import COST_PLOT_STYLE
from neuralls.domain.solver.models.result import COST_FIELD_REGISTRY


def test_plot_styles_cover_exactly_the_registered_cost_fields() -> None:
    assert set(COST_PLOT_STYLE) == {spec.field_name for spec in COST_FIELD_REGISTRY}


def test_filename_suffixes_are_unique() -> None:
    suffixes = [style.filename_suffix for style in COST_PLOT_STYLE.values()]
    assert len(suffixes) == len(set(suffixes))
