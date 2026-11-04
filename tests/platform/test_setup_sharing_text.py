"""The text an online figure draws about its shared setup, read from the record on its card."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.solver.models.figure_scope import AllDatasetsScope
from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.platform.reporting.figure_evidence import (
    NoSharedSetup,
    SetupSharedOverRhs,
    SetupSharing,
)
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.figure_text import ScopeSubtitle, setup_sharing_note
from neuralls.platform.reporting.group_plots import GroupStyler
from neuralls.platform.reporting.outcome_plots import GroupStyle
from neuralls.shared.scientific_description.policy import NotationPolicy

SHARED = "each matrix's setup is shared by its r_m planned RHS"


@pytest.mark.parametrize(
    ("sharing", "expected"),
    [
        (NoSharedSetup(), None),
        (SetupSharedOverRhs((4,)), f"{SHARED}, r_m = 4"),
        (SetupSharedOverRhs((2, 4)), f"{SHARED}, r_m = 2 or 4 (varies by matrix)"),
        (SetupSharedOverRhs((2, 4, 8)), f"{SHARED}, r_m = 2, 4 or 8 (varies by matrix)"),
        (
            SetupSharedOverRhs((1, 2, 3, 5, 9)),
            f"{SHARED}, r_m from 1 to 9 over 5 values (varies by matrix)",
        ),
        (SetupSharedOverRhs(()), SHARED),
    ],
)
def test_the_note_states_r_m_as_recorded(sharing: SetupSharing, expected: str | None) -> None:
    assert setup_sharing_note(sharing) == expected


@pytest.mark.parametrize(
    ("sharing", "note"),
    [(NoSharedSetup(), None), (SetupSharedOverRhs((4,)), f"{SHARED}, r_m = 4")],
)
def test_the_subtitle_adds_the_note_under_the_scope(
    group_style_of: Callable[..., GroupStyle],
    notation: NotationPolicy,
    sharing: SetupSharing,
    note: str | None,
) -> None:
    style = group_style_of("amg", setup_sharing=sharing)
    scope = ScopeSubtitle(AllDatasetsScope(), notation).text

    assert style.subtitle == (scope if note is None else f"{scope}\n{note}")


def test_every_figure_is_styled_with_the_setup_sharing_its_card_records(
    profile_report: RecipeProfileReport,
    repeated_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
) -> None:
    styler = GroupStyler(repeated_timing_presentation, notation)
    plans = figure_plans(profile_report, repeated_timing_presentation, "key", {})

    assert [styler.style(plan).setup_sharing for plan in plans] == [
        plan.metadata.setup_sharing for plan in plans
    ]
