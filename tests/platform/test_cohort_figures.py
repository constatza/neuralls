"""The figures planned at each report level, and what a pooled mixture figure draws and stores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.environment_support import SESSION_ENVIRONMENT
from tests.figure_support import no_offline_cost
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.solver.models.figure_scope import AllDatasetsScope
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.labels import AllMatrixSetsLabel
from neuralls.domain.solver.reporting.cohorts import MixtureCohort
from neuralls.platform.config.models.figure_style import FigureStyleConfig
from neuralls.platform.reporting import group_plots
from neuralls.platform.reporting.figure_card import (
    build_figure_card,
    card_payload,
    decode_figure_card,
)
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_metadata import (
    FigureKind,
    FigurePlan,
    GroupFigureMetadata,
    SiblingTable,
    TableKind,
)
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.figure_recipes import group_recipe
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.platform.reporting.group_drawers import FigureData
from neuralls.platform.reporting.group_plots import GroupStyler, write_outcome_plots
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.scientific_description.policy import NotationPolicy

KEY = "key"

_RAW_SCALE = (
    "workload_solve_bars",
    "workload_online_bars",
    "workload_online_amortized_bars",
    "full_cost_breakdown",
    "breakeven_online_amortized",
    "breakeven_full",
    "matrix_heatmap_iterations",
    "convergence_factor",
    "iteration_means",
    "time_means_solve_own",
    "time_means_solve_common",
    "time_means_online_amortized_own",
    "time_means_online_amortized_common",
    "time_means_online_own",
    "time_means_online_common",
    "setup_means",
    "time_per_iteration",
    "iteration_distribution",
    "time_distribution_solve",
    "time_distribution_online",
    "time_distribution_online_amortized",
)
"""Raw-scale figures, drawn only at the matrix-set-and-RHS-type level."""

_POOLED_FIGURES = (
    "paired_online_scatter",
    "paired_online_amortized_scatter",
    "speed_ratio_online_cdf",
    "speed_ratio_online_amortized_cdf",
    "performance_profile_solve",
    "performance_profile_online",
    "performance_profile_online_amortized",
    "ecdf_iterations",
    "ecdf_solve",
    "ecdf_online",
    "ecdf_online_amortized",
)
"""Scale-free figures: one unsuffixed figure at recipe level, one pooled figure at mixtures."""


def _stems(*names: str, suffix: str = "") -> set[str]:
    return {f"{name}_{KEY}{suffix}" for name in names}


@pytest.fixture
def presentation() -> GroupPresentation:
    return GroupPresentation(scope=AllDatasetsScope(), plot_styles={}, methods={})


@pytest.fixture
def all_sets_report(two_recipe_mixture_report: WorkloadMixtureReport) -> WorkloadMixtureReport:
    """The two-recipe workload pooled over every matrix set."""
    return replace(two_recipe_mixture_report, label=AllMatrixSetsLabel((GAUSSIAN, SINE)))


def _planned(report: GroupReport, presentation: GroupPresentation) -> list[FigurePlan]:
    return list(figure_plans(report, presentation, KEY, {}))


def _stems_of(plans: list[FigurePlan]) -> set[str]:
    return {plan.stem for plan in plans}


def test_a_matrix_set_and_rhs_type_report_has_one_unsuffixed_figure_of_every_kind(
    profile_report: RecipeProfileReport, presentation: GroupPresentation
) -> None:
    assert _stems_of(_planned(profile_report, presentation)) == _stems(
        *_RAW_SCALE, *_POOLED_FIGURES
    )


@pytest.mark.parametrize("level", ["matrix_set", "all_matrix_sets"])
def test_a_mixture_report_has_only_the_pooled_scale_free_figures(
    two_recipe_mixture_report: WorkloadMixtureReport,
    all_sets_report: WorkloadMixtureReport,
    presentation: GroupPresentation,
    level: str,
) -> None:
    report = two_recipe_mixture_report if level == "matrix_set" else all_sets_report

    assert _stems_of(_planned(report, presentation)) == _stems(*_POOLED_FIGURES, suffix="__mixture")


@dataclass(frozen=True, slots=True)
class Drawn:
    """What one figure's drawer was handed."""

    stem: str
    data: FigureData


@pytest.fixture
def drawn(monkeypatch: pytest.MonkeyPatch) -> list[Drawn]:
    """Replace every recipe's drawer by a recorder, so no figure is rendered."""
    calls: list[Drawn] = []

    def recording_recipe(kind: FigureKind):
        def draw(data: FigureData, target: FigureTarget) -> FigureTarget:
            calls.append(Drawn(target.stem, data))
            return target

        return replace(group_recipe(kind), draw=draw)

    monkeypatch.setattr(group_plots, "group_recipe", recording_recipe)
    return calls


@pytest.fixture
def write_all(
    all_sets_report: WorkloadMixtureReport,
    presentation: GroupPresentation,
    notation: NotationPolicy,
    tmp_path: Path,
    drawn: list[Drawn],
) -> Callable[[], tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]]:
    def write() -> tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]:
        plans = figure_plans(all_sets_report, presentation, KEY, {})
        staged = write_outcome_plots(
            all_sets_report,
            figure_summaries(all_sets_report, no_offline_cost(all_sets_report)),
            GroupStyler(presentation, notation, FigureStyleConfig()),
            tmp_path,
            plans,
        )
        return (
            {plan.stem: plan.metadata for plan in plans},
            {figure.figure.plan.stem: figure.table for figure in staged},
        )

    return write


_MIXTURE_NOTE = "equal-weight mixture of Gaussian, Sinusoidal ($\\omega=6.28319$)"

_SETUP_SHARING_LINE = (
    "each matrix's setup is shared by its r_m planned RHS, r_m = 1, 2 or 3 (varies by matrix)"
)


@pytest.mark.parametrize(
    ("stem", "extra_lines"),
    [
        ("ecdf_iterations", []),
        ("performance_profile_solve", []),
        ("speed_ratio_online_cdf", []),
        ("speed_ratio_online_amortized_cdf", [_SETUP_SHARING_LINE]),
    ],
)
def test_a_pooled_profile_figure_is_a_mixture_figure_drawn_whole(
    all_sets_report: WorkloadMixtureReport,
    drawn: list[Drawn],
    write_all: Callable[[], tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]],
    stem: str,
    extra_lines: list[str],
) -> None:
    metadata, tables = write_all()
    full = figure_summaries(all_sets_report, no_offline_cost(all_sets_report))
    pooled = f"{stem}_{KEY}__mixture"

    (call,) = (call for call in drawn if call.stem == pooled)

    assert metadata[pooled].cohort == MixtureCohort()
    assert_frame_equal(tables[pooled], full[stem])
    assert_frame_equal(call.data.observations, all_sets_report.observations)
    assert call.data.style.subtitle.splitlines() == [
        "All comparison matrix sets",
        _MIXTURE_NOTE,
        *extra_lines,
    ]


def test_a_pooled_figures_card_names_the_equal_weight_mixture_and_round_trips(
    all_sets_report: WorkloadMixtureReport,
    presentation: GroupPresentation,
    notation: NotationPolicy,
) -> None:
    (plan,) = (
        plan
        for plan in _planned(all_sets_report, presentation)
        if plan.stem == f"ecdf_iterations_{KEY}__mixture"
    )

    card = build_figure_card(
        plan.stem,
        plan.title,
        plan.metadata,
        session_run_id="session",
        environment=SESSION_ENVIRONMENT,
        notation=notation,
    )
    payload = card_payload(card)
    decoded = decode_figure_card(payload)

    assert payload["cohort"] == (
        f"equal-weight mixture of {recipe_key(GAUSSIAN)}, {recipe_key(SINE)}"
    )
    assert decoded == card
    assert decoded.table == SiblingTable(f"{plan.stem}.parquet", TableKind.SERIES)
