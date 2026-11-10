"""How a report group's preconditioners are titled and drawn, shared by every group plot."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

from neuralls.domain.solver.models.figure_scope import FigureScope
from neuralls.platform.reporting.figure_evidence import SetupSharing
from neuralls.platform.reporting.figure_text import Legend, ScopeSubtitle, setup_sharing_note
from neuralls.shared.scientific_description.policy import NotationPolicy


@dataclass(frozen=True, slots=True)
class GroupStyle:
    """How a group's preconditioners are titled and drawn, shared by every plot here.

    ``title`` is fixed editorial wording; the subtitle is the typed ``scope`` rendered
    when a figure is drawn, followed by what ``setup_sharing`` (the record the figure's card
    stores) says about the setup the metric divides, and every series is named by ``legend``.
    """

    title: str
    scope: FigureScope
    setup_sharing: SetupSharing
    notation: NotationPolicy
    legend: Legend
    families: Mapping[str, Hashable]
    color_keys: Mapping[str, Hashable]
    marker_keys: Mapping[str, Hashable]

    @property
    def subtitle(self) -> str:
        """The scope under the title, and the shared setup when the metric divides one."""
        scope = ScopeSubtitle(self.scope, self.notation).text
        match setup_sharing_note(self.setup_sharing):
            case None:
                return scope
            case note:
                return f"{scope}\n{note}"
