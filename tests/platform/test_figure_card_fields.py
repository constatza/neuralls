"""The fields a group figure's card records about the statistic it draws, present and typed.

A card is built the way the report sink builds it: the figure plan of a group report goes
through ``build_figure_card`` and ``card_payload``. ``metric``, ``population`` and ``statistic`` name
what is summarized. The payload is read by key:

- ``interval`` is a table with ``method``, ``level``, ``resamples`` and ``matrices``, present only
  on a figure that draws one;
- ``r_m`` lists the distinct planned RHS counts per matrix, present only on online metrics;
- ``outcome_counts`` holds, per method, the number of systems with each outcome;
- ``timing_spread`` holds, per method, the median relative spread of its timing repeats, and
  ``timing_warning`` is present only when a figure rests on a single timing repeat; both exist
  only on a figure of a metric measured in seconds, since iteration counts do not depend on timing.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import pytest
from tests.figure_support import VARIANT_METRIC

from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    paired_systems,
    speed_ratios,
)
from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.domain.solver.reporting.summary import Population, select, timing_noise
from neuralls.platform.reporting.figure_card import (
    build_figure_card,
    card_payload,
    decode_figure_card,
)
from neuralls.platform.reporting.figure_metadata import GroupFigureMetadata, Statistic
from neuralls.platform.reporting.figure_planning import (
    FIGURE_RECIPES,
    GroupFigureRecipe,
    figure_plans,
)
from neuralls.shared.constants import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from neuralls.shared.digest import Json
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import PreconditionerName

type Payload = Mapping[str, Json]

FILE_KEY = "key"
SESSION_RUN_ID = "session-run"
INTERVAL_METHOD = "matrix_cluster_bootstrap"

_GROUP_RECIPES = [
    recipe for recipe in FIGURE_RECIPES.values() if isinstance(recipe, GroupFigureRecipe)
]
_PROFILE_VARIANTS = [
    variant
    for recipe in _GROUP_RECIPES
    if RecipeProfileReport in recipe.applies_to
    for variant in recipe.variants
]
_BY_STEM = {variant.stem: variant for variant in _PROFILE_VARIANTS}
_RECIPE_OF_STEM = {variant.stem: recipe for recipe in _GROUP_RECIPES for variant in recipe.variants}
_ONLINE_STEMS = sorted(
    variant.stem for variant in _PROFILE_VARIANTS if variant.metric is MetricId.ONLINE_S
)
_NOT_ONLINE_STEMS = sorted(
    variant.stem for variant in _PROFILE_VARIANTS if variant.metric is not MetricId.ONLINE_S
)
_MEAN_STEMS = sorted(
    variant.stem
    for variant in _PROFILE_VARIANTS
    if variant.statistic in {Statistic.MEAN_OVER_SYSTEMS, Statistic.MEAN_OVER_MATRICES}
)
_OWN_MEAN_STEMS = [stem for stem in _MEAN_STEMS if _BY_STEM[stem].population is Population.OWN]
_COMMON_MEAN_STEMS = [stem for stem in _MEAN_STEMS if stem not in _OWN_MEAN_STEMS]
_NO_INTERVAL_STEMS = sorted(set(_BY_STEM) - {"speed_ratio_cdf", *_MEAN_STEMS})

_TIME_STEMS = sorted(
    variant.stem for variant in _PROFILE_VARIANTS if variant.metric is not MetricId.ITERATIONS
)
_ITERATION_STEMS = sorted(
    variant.stem for variant in _PROFILE_VARIANTS if variant.metric is MetricId.ITERATIONS
)


def _payloads(
    report: RecipeProfileReport, presentation: GroupPresentation, notation: NotationPolicy
) -> dict[str, Payload]:
    """The card payload of every figure planned for ``report``, by variant stem."""
    stems = {variant.filename(FILE_KEY): variant.stem for variant in _PROFILE_VARIANTS}
    return {
        stems[plan.filename]: card_payload(
            build_figure_card(
                plan.filename,
                plan.title,
                plan.metadata,
                session_run_id=SESSION_RUN_ID,
                notation=notation,
            )
        )
        for plan in figure_plans(report, presentation, FILE_KEY, {})
    }


@pytest.fixture
def payloads(
    profile_report: RecipeProfileReport,
    repeated_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
) -> dict[str, Payload]:
    """The cards of a report whose systems were each timed three times."""
    return _payloads(profile_report, repeated_timing_presentation, notation)


@pytest.fixture
def single_repeat_payloads(
    profile_report: RecipeProfileReport,
    single_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
) -> dict[str, Payload]:
    """The cards of a report whose systems were each timed once."""
    return _payloads(profile_report, single_timing_presentation, notation)


def _leaf_types(value: object) -> set[type]:
    """The types of the scalars nested anywhere in a card value."""
    match value:
        case Mapping():
            return set().union(*(_leaf_types(item) for item in value.values()))
        case list():
            return set().union(*(_leaf_types(item) for item in value))
        case scalar:
            return {type(scalar)}


def _table(payload: Payload, key: str) -> Mapping[str, object]:
    match payload.get(key):
        case dict() as table:
            return table
        case other:
            raise AssertionError(f"the card has no {key} table, it has {other!r}")


@pytest.mark.parametrize("stem", sorted(_BY_STEM))
def test_a_card_records_the_failure_treatment_of_its_recipe_as_a_typed_value(
    payloads: dict[str, Payload], stem: str
) -> None:
    expected = _RECIPE_OF_STEM[stem].failure_treatment

    match decode_figure_card(payloads[stem]).metadata:
        case GroupFigureMetadata(failure_treatment=treatment):
            assert treatment is expected
        case other:
            raise AssertionError(f"a group figure decodes to group metadata, not {other!r}")
    assert payloads[stem]["failure_treatment"] == expected.value


@pytest.mark.parametrize("stem", _NO_INTERVAL_STEMS)
def test_a_figure_without_an_interval_records_none(payloads: dict[str, Payload], stem: str) -> None:
    assert "interval" not in payloads[stem]


@pytest.mark.parametrize("stem", _NOT_ONLINE_STEMS)
def test_a_figure_of_a_metric_without_a_shared_setup_records_no_r_m(
    payloads: dict[str, Payload], stem: str
) -> None:
    assert "r_m" not in payloads[stem]


def test_a_card_records_no_timing_warning_when_systems_were_timed_repeatedly(
    payloads: dict[str, Payload],
) -> None:
    assert [stem for stem, payload in payloads.items() if "timing_warning" in payload] == []


@pytest.mark.parametrize("stem", sorted(VARIANT_METRIC))
def test_a_card_names_the_metric_it_summarizes(payloads: dict[str, Payload], stem: str) -> None:
    assert payloads[stem].get("metric") == VARIANT_METRIC[stem].value


@pytest.mark.parametrize("stem", sorted(VARIANT_METRIC))
def test_a_card_names_the_population_of_its_statistic(
    payloads: dict[str, Payload], stem: str
) -> None:
    assert payloads[stem].get("population") in {population.value for population in Population}


@pytest.mark.parametrize("stem", sorted(_BY_STEM))
def test_a_card_names_the_statistic_it_draws(payloads: dict[str, Payload], stem: str) -> None:
    assert isinstance(payloads[stem].get("statistic"), str)


def test_the_speed_ratio_card_states_its_interval(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload]
) -> None:
    ratios = speed_ratios(
        paired_systems(profile_report.observations, PreconditionerName("jacobi"), REFERENCE_METRIC)
    )
    matrices = ratios.select(pl.struct("dataset", "matrix_id").n_unique()).item()

    interval = _table(payloads["speed_ratio_cdf"], "interval")

    assert interval == {
        "method": INTERVAL_METHOD,
        "level": CONFIDENCE_LEVEL,
        "resamples": BOOTSTRAP_RESAMPLES,
        "matrices": matrices,
    }
    assert {key: type(value) for key, value in interval.items()} == {
        "method": str,
        "level": float,
        "resamples": int,
        "matrices": int,
    }


def test_mean_figures_exist_for_every_mean_kind() -> None:
    assert {_RECIPE_OF_STEM[stem].kind.value for stem in _MEAN_STEMS} == {
        "iteration_means",
        "time_means",
        "setup_means",
    }


@pytest.mark.parametrize("stem", _OWN_MEAN_STEMS)
def test_a_mean_card_states_the_matrix_cluster_bootstrap_of_its_whiskers(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload], stem: str
) -> None:
    variant = _BY_STEM[stem]
    selected = select(profile_report.observations, variant.metric, variant.population)
    matrices = selected.select(pl.struct("dataset", "matrix_id").n_unique()).item()

    assert _table(payloads[stem], "interval") == {
        "method": INTERVAL_METHOD,
        "level": CONFIDENCE_LEVEL,
        "resamples": BOOTSTRAP_RESAMPLES,
        "matrices": matrices,
    }


@pytest.mark.parametrize("stem", _COMMON_MEAN_STEMS)
def test_a_mean_card_over_an_empty_common_population_records_no_interval(
    payloads: dict[str, Payload], stem: str
) -> None:
    # ``broken`` defines nothing, so no system is common to every method and no whisker exists.
    assert "interval" not in payloads[stem]


@pytest.mark.parametrize("stem", _ONLINE_STEMS)
def test_an_online_card_lists_the_planned_rhs_counts_per_matrix(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload], stem: str
) -> None:
    per_matrix = profile_report.observations.unique(["dataset", "matrix_id"])["planned_rhs"]

    r_m = payloads[stem].get("r_m")

    assert r_m == sorted(per_matrix.unique().to_list())
    assert _leaf_types(r_m) == {int}


@pytest.mark.parametrize("stem", sorted(_BY_STEM))
def test_a_card_counts_the_systems_of_each_method_by_outcome(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload], stem: str
) -> None:
    observations = profile_report.observations
    expected = {
        method: {
            outcome.value: observations.filter(
                (pl.col("method") == method) & (pl.col("outcome") == outcome.value)
            ).height
            for outcome in Outcome
        }
        for method in sorted(observations["method"].unique())
    }

    counts = _table(payloads[stem], "outcome_counts")

    assert counts == expected
    assert _leaf_types(counts) == {int}


@pytest.mark.parametrize("stem", _TIME_STEMS)
def test_a_card_records_the_median_timing_spread_of_each_method_with_timed_solves(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload], stem: str
) -> None:
    expected = {
        row["method"]: row["median_timing_spread"]
        for row in timing_noise(profile_report.observations).iter_rows(named=True)
        if row["median_timing_spread"] is not None
    }

    spread = _table(payloads[stem], "timing_spread")

    assert spread == expected
    assert _leaf_types(spread) == {float}


@pytest.mark.parametrize("stem", _ITERATION_STEMS)
def test_a_card_of_a_metric_that_is_not_a_time_records_no_timing_evidence(
    single_repeat_payloads: dict[str, Payload], stem: str
) -> None:
    payload = single_repeat_payloads[stem]

    assert ("timing_spread" in payload, "timing_warning" in payload) == (False, False)


@pytest.mark.parametrize("stem", _BY_STEM)
def test_a_card_decodes_to_the_metadata_it_was_built_from(
    profile_report: RecipeProfileReport,
    single_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
    stem: str,
) -> None:
    plans = {
        plan.filename: plan
        for plan in figure_plans(profile_report, single_timing_presentation, FILE_KEY, {})
    }
    plan = plans[_BY_STEM[stem].filename(FILE_KEY)]
    card = build_figure_card(
        plan.filename, plan.title, plan.metadata, session_run_id=SESSION_RUN_ID, notation=notation
    )

    assert decode_figure_card(card_payload(card)) == card


@pytest.mark.parametrize("stem", _TIME_STEMS)
def test_a_card_resting_on_a_single_timing_repeat_carries_a_warning(
    single_repeat_payloads: dict[str, Payload], stem: str
) -> None:
    assert isinstance(single_repeat_payloads[stem].get("timing_warning"), str)
