"""How a report group's preconditioners are titled and drawn, shared by every group plot."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

from neuralls.domain.solver.models.figure_scope import FigureScope
from neuralls.domain.solver.reporting.cohorts import Cohort
from neuralls.platform.reporting.distribution_mode import RowLimits
from neuralls.platform.reporting.figure_evidence import SetupSharing
from neuralls.platform.reporting.figure_text import (
    Legend,
    ScopeSubtitle,
    cohort_note,
    setup_sharing_note,
)
from neuralls.shared.scientific_description.policy import NotationPolicy


@dataclass(frozen=True, slots=True)
class GroupStyle:
    """How a group's preconditioners are titled and drawn, shared by every plot here.

    ``title`` is fixed editorial wording; the subtitle is the typed ``scope`` rendered
    when a figure is drawn, followed by the recipe or mixture a cohort figure draws and by what
    ``setup_sharing`` (the record the figure's card stores) says about the setup the metric
    divides, and every series is named by ``legend``. ``label_rotation_deg`` is the configured
    rotation of x-axis labels and ``row_limits`` the sample sizes from which
    a distribution row gains a box and a violin. ``show_titles`` off draws the figure without
    its title and subtitle.
    """

    title: str
    scope: FigureScope
    cohort: Cohort
    setup_sharing: SetupSharing
    notation: NotationPolicy
    legend: Legend
    families: Mapping[str, Hashable]
    color_keys: Mapping[str, Hashable]
    marker_keys: Mapping[str, Hashable]
    label_rotation_deg: float
    row_limits: RowLimits
    show_titles: bool

    @property
    def subtitle(self) -> str:
        """The scope under the title, the cohort drawn, and the shared setup the metric divides."""
        notes = (
            cohort_note(self.cohort, self.notation),
            setup_sharing_note(self.setup_sharing),
        )
        return "\n".join(
            [ScopeSubtitle(self.scope, self.notation).text, *(note for note in notes if note)]
        )
