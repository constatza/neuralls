"""The fields a group figure's card records about the statistic it draws, present and typed.

A card is built the way the report sink builds it: the figure plan of a group report goes
through ``build_figure_card`` and ``card_payload``. ``metric``, ``population`` and ``statistic`` name
what is summarized. The payload is read by key:

- ``interval`` is a table on every card whose ``kind`` is ``matrix_cluster_bootstrap`` (with
  ``level``, ``resamples`` and ``matrices``), ``too_few_matrices`` (``matrices``, ``minimum``) or
  ``not_drawn_by_kind``;
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
from tests.environment_support import SESSION_ENVIRONMENT
from tests.figure_support import VARIANT_METRIC

from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    speed_ratio_summary,
)
from neuralls.domain.benchmarking.profiles import performance_series
from neuralls.domain.benchmarking.uncertainty import MIN_INTERVAL_MATRICES, TooFewMatrices
from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId, Unit
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.domain.solver.reporting.summary import Population, summarize, timing_noise
from neuralls.platform.reporting.figure_card import (
    InvalidFigureCardError,
    build_figure_card,
    card_payload,
    decode_figure_card,
)
from neuralls.platform.reporting.figure_evidence import (
    intervals_of_means,
    intervals_of_speed_ratios,
)
from neuralls.platform.reporting.figure_metadata import GroupFigureMetadata, Statistic
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.figure_recipes import FIGURE_RECIPES, GroupFigureRecipe
from neuralls.shared.constants import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from neuralls.shared.digest import Json
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import PreconditionerName

type Payload = Mapping[str, Json]

FILE_KEY = "key"
SESSION_RUN_ID = "session-run"
BOOTSTRAP_KIND = "matrix_cluster_bootstrap"

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
_NO_INTERVAL_STEMS = sorted(set(_BY_STEM) - {"speed_ratio_cdf", "convergence_factor", *_MEAN_STEMS})
_WHISKER_STEMS = [*_OWN_MEAN_STEMS, "convergence_factor"]
"""Figures whose whiskers are matrix-cluster intervals of per-method means over their own systems."""

_TIME_UNITS = frozenset({Unit.SECONDS, Unit.SECONDS_PER_ITERATION})
_TIME_STEMS = sorted(
    variant.stem for variant in _PROFILE_VARIANTS if METRICS[variant.metric].unit in _TIME_UNITS
)
_UNTIMED_STEMS = sorted(set(_BY_STEM) - set(_TIME_STEMS))


def _payloads(
    report: RecipeProfileReport, presentation: GroupPresentation, notation: NotationPolicy
) -> dict[str, Payload]:
    """The card payload of every figure planned for ``report``, by variant stem."""
    stems = {variant.figure_stem(FILE_KEY): variant.stem for variant in _PROFILE_VARIANTS}
    return {
        stems[plan.stem]: card_payload(
            build_figure_card(
                plan.stem,
                plan.title,
                plan.metadata,
                session_run_id=SESSION_RUN_ID,
                environment=SESSION_ENVIRONMENT,
                notation=notation,
            )
        )
        for plan in figure_plans(report, presentation, FILE_KEY, {})
    }


@pytest.fixture(scope="module")
def payloads(
    profile_report: RecipeProfileReport,
    repeated_timing_presentation: GroupPresentation,
    notation: NotationPolicy,
) -> dict[str, Payload]:
    """The cards of a report whose systems were each timed three times."""
    return _payloads(profile_report, repeated_timing_presentation, notation)


@pytest.fixture(scope="module")
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
def test_a_figure_whose_kind_draws_no_interval_says_so(
    payloads: dict[str, Payload], stem: str
) -> None:
    assert _table(payloads[stem], "interval") == {"kind": "not_drawn_by_kind"}


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


def test_the_cost_breakdown_card_names_its_statistic_table_and_failure_rule(
    payloads: dict[str, Payload],
) -> None:
    card = payloads["cost_breakdown"]

    assert (
        card["kind"],
        card["statistic"],
        card["failure_treatment"],
        card["table"],
        card["table_kind"],
    ) == (
        "cost_breakdown",
        "offline_online_total",
        "none",
        "cost_breakdown_key.parquet",
        "cost_breakdown",
    )


def test_the_speed_ratio_card_states_each_methods_interval_as_its_table_counts_it(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload]
) -> None:
    summary = speed_ratio_summary(
        profile_report.observations, PreconditionerName("jacobi"), REFERENCE_METRIC
    )

    assert _method_matrices(payloads["speed_ratio_cdf"]) == _expected_matrices(summary)


def test_a_drawn_interval_entry_states_its_bootstrap_with_toml_types(
    payloads: dict[str, Payload],
) -> None:
    drawn = [
        entry
        for entry in _method_entries(payloads["speed_ratio_cdf"])
        if entry["kind"] == BOOTSTRAP_KIND
    ]

    assert {(entry["level"], entry["resamples"], type(entry["matrices"])) for entry in drawn} == {
        (CONFIDENCE_LEVEL, BOOTSTRAP_RESAMPLES, int)
    }


def test_mean_figures_exist_for_every_mean_kind() -> None:
    assert {_RECIPE_OF_STEM[stem].kind.value for stem in _MEAN_STEMS} == {
        "iteration_means",
        "time_means",
        "setup_means",
    }


@pytest.mark.parametrize(
    ("stem", "definition"),
    [
        (
            "iteration_distribution",
            (
                "distribution of iterations to converge over systems, own population; box at "
                "q25, q50, q75 and whiskers from q05 to q95 of the systems, a distribution "
                "summary and not a confidence interval; points are per-matrix means"
            ),
        ),
        (
            "time_distribution_online",
            (
                "distribution of amortized online time per system [s] over systems, own population; box at "
                "q25, q50, q75 and whiskers from q05 to q95 of the systems, a distribution "
                "summary and not a confidence interval; points are per-matrix means"
            ),
        ),
    ],
)
def test_a_distribution_card_states_its_whiskers_are_no_confidence_interval(
    payloads: dict[str, Payload], stem: str, definition: str
) -> None:
    assert (payloads[stem]["metric_definition"], _table(payloads[stem], "interval")) == (
        definition,
        {"kind": "not_drawn_by_kind"},
    )


def test_the_matrix_heatmap_card_states_what_a_cell_is() -> None:
    assert _BY_STEM["matrix_heatmap_iterations"].metric_definition == (
        "mean of iterations to converge over the converged RHS of each matrix, per method; "
        "a hatched cell has no converged RHS and is labelled 0/r solved, r the planned RHS "
        "of the matrix"
    )


@pytest.mark.parametrize("stem", _MEAN_STEMS)
def test_a_mean_card_states_each_methods_interval_as_its_table_counts_it(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload], stem: str
) -> None:
    variant = _BY_STEM[stem]
    summary = summarize(profile_report.observations, variant.metric, variant.population)

    assert _method_matrices(payloads[stem]) == _expected_matrices(summary)


@pytest.mark.parametrize("stem", _COMMON_MEAN_STEMS)
def test_a_mean_card_over_an_empty_common_population_states_too_few_matrices(
    payloads: dict[str, Payload], stem: str
) -> None:
    # ``broken`` defines nothing, so no system is common to every method and no whisker exists.
    assert {
        (entry["kind"], entry["matrices"], entry["minimum"])
        for entry in _method_entries(payloads[stem])
    } == {("too_few_matrices", 0, MIN_INTERVAL_MATRICES)}


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


@pytest.mark.parametrize("stem", sorted(_BY_STEM))
def test_a_card_records_the_systems_a_performance_profile_left_out(
    profile_report: RecipeProfileReport, payloads: dict[str, Payload], stem: str
) -> None:
    payload = payloads[stem]
    expected = (
        performance_series(
            profile_report.observations, MetricId(str(payload["metric"]))
        ).n_excluded_zero_best
        if payload["statistic"] == Statistic.PERFORMANCE_PROFILE.value
        else 0
    )

    assert payload["excluded_zero_best"] == expected


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


@pytest.mark.parametrize("stem", _UNTIMED_STEMS)
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
        plan.stem: plan
        for plan in figure_plans(profile_report, single_timing_presentation, FILE_KEY, {})
    }
    plan = plans[_BY_STEM[stem].figure_stem(FILE_KEY)]
    card = build_figure_card(
        plan.stem,
        plan.title,
        plan.metadata,
        session_run_id=SESSION_RUN_ID,
        environment=SESSION_ENVIRONMENT,
        notation=notation,
    )

    assert decode_figure_card(card_payload(card)) == card


@pytest.mark.parametrize("stem", _TIME_STEMS)
def test_a_card_resting_on_a_single_timing_repeat_carries_a_warning(
    single_repeat_payloads: dict[str, Payload], stem: str
) -> None:
    assert isinstance(single_repeat_payloads[stem].get("timing_warning"), str)


@pytest.fixture
def one_matrix_observations(profile_report: RecipeProfileReport) -> pl.DataFrame:
    """The systems of the first matrix of ``profile_report`` only."""
    first = profile_report.observations["matrix_id"].min()
    return profile_report.observations.filter(pl.col("matrix_id") == first)


def test_a_speed_ratio_over_one_matrix_has_too_few_matrices_for_an_interval(
    one_matrix_observations: pl.DataFrame,
) -> None:
    intervals = intervals_of_speed_ratios(
        one_matrix_observations, PreconditionerName("jacobi"), REFERENCE_METRIC
    )

    assert {entry.method: entry.interval for entry in intervals.methods} == {
        PreconditionerName("amg"): TooFewMatrices(1),
        PreconditionerName("broken"): TooFewMatrices(0),
    }


def test_means_over_one_matrix_have_too_few_matrices_for_an_interval(
    one_matrix_observations: pl.DataFrame,
) -> None:
    intervals = intervals_of_means(one_matrix_observations, MetricId.ITERATIONS, Population.OWN)

    assert {entry.method: entry.interval for entry in intervals.methods} == {
        PreconditionerName("amg"): TooFewMatrices(1),
        PreconditionerName("broken"): TooFewMatrices(0),
        PreconditionerName("jacobi"): TooFewMatrices(1),
    }


def _per_method(*entries: object) -> dict[str, object]:
    return {"kind": "per_method", "methods": list(entries)}


_BOOT = {"method": "amg", "kind": "matrix_cluster_bootstrap", "level": 0.95, "resamples": 10}


@pytest.mark.parametrize(
    "interval",
    [
        {"kind": "normal_approximation"},
        {"kind": "not_drawn_by_kind", "matrices": 5},
        {"kind": "per_method"},
        _per_method({"method": "amg", "kind": "too_few_matrices", "matrices": 1}),
        _per_method({"method": "amg", "kind": "too_few_matrices", "matrices": 5, "minimum": 2}),
        _per_method({**_BOOT, "matrices": 1}),
        _per_method({**_BOOT, "matrices": "4"}),
        _per_method({**_BOOT, "matrices": True}),
        _per_method({"method": "amg", "kind": "too_few_matrices", "matrices": 1, "minimum": 3}),
        _per_method({**_BOOT, "matrices": 4}, {**_BOOT, "matrices": 4}),
        _per_method({**_BOOT, "method": 7, "matrices": 4}),
        _per_method({"method": "amg", "kind": "too_few_matrices", "matrices": -1, "minimum": 2}),
        _per_method({**_BOOT, "matrices": 4, "level": 5.0}),
        _per_method({**_BOOT, "matrices": 4, "resamples": 0}),
        "none",
    ],
    ids=[
        "unknown kind",
        "extra field",
        "no methods",
        "missing minimum",
        "too few with enough matrices",
        "bootstrap over one matrix",
        "matrices as text",
        "matrices as bool",
        "another minimum",
        "a method twice",
        "method not a string",
        "negative matrices",
        "level above one",
        "no resamples",
        "not a table",
    ],
)
def test_a_card_with_a_malformed_interval_is_rejected(
    payloads: dict[str, Payload], interval: object
) -> None:
    payload = {**payloads["speed_ratio_cdf"], "interval": interval}

    with pytest.raises(InvalidFigureCardError):
        decode_figure_card(payload)


def _method_entries(payload: Payload) -> list[Mapping[str, object]]:
    interval = _table(payload, "interval")
    assert interval["kind"] == "per_method"
    match interval["methods"]:
        case list() as entries:
            return entries
        case other:
            raise AssertionError(f"interval methods is not an array of tables: {other!r}")


def _method_matrices(payload: Payload) -> dict[object, tuple[object, object]]:
    return {
        entry["method"]: (entry["kind"], entry["matrices"]) for entry in _method_entries(payload)
    }


def _expected_matrices(summary: pl.DataFrame) -> dict[object, tuple[object, object]]:
    """Each method's interval kind and matrices as its row of the figure's table counts them."""
    return {
        method: (
            BOOTSTRAP_KIND if matrices >= MIN_INTERVAL_MATRICES else "too_few_matrices",
            matrices,
        )
        for method, matrices in summary.select("method", "n_matrices").iter_rows()
    }
