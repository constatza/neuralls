from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceUnavailable,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.figure_scope import (
    AllDatasetsScope,
    DatasetScope,
    FigureScope,
    MethodScope,
    RecipeScope,
    SystemScope,
    render_figure_scope,
)
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.figure_card import (
    FigureCard,
    InvalidFigureCardError,
    build_figure_card,
    card_name,
    card_payload,
    read_figure_card,
    validate_figure_card,
    write_figure_card,
)
from neuralls.platform.reporting.figure_metadata import (
    CARD_PROFILE,
    MAX_SLUG_LENGTH,
    FigureKind,
    FigureMetadata,
    FigureMethod,
    FigurePlan,
    FigureSlugs,
    GroupFigureMetadata,
    MethodFigureMetadata,
    StagedFigure,
    collision_key,
)
from neuralls.platform.storage.facts_codec import decode_figure_scope, encode_figure_scope
from neuralls.shared.rhs_recipe import SinusoidalRecipe
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import MatrixId, PreconditionerName

SESSION_RUN_ID = "session-run"
FIGURE = "performance_profile_fine_x_online.png"
CONTEXT = SolverTimingContext(
    rtol=1e-6,
    atol=1e-14,
    max_iterations=100,
    stopping_criterion="residual_norm",
    m_max=5,
    reference_precision_margin=2.0,
    timing_repeats=3,
    timing_warmup=1,
    timing_statistic="median",
    normalization="matrix",
)


@pytest.fixture
def scope(dataset_presentation: DatasetPresentation) -> RecipeScope:
    return RecipeScope(dataset_presentation, SinusoidalRecipe(omega=6.28319))


def _card(metadata: FigureMetadata, notation: NotationPolicy) -> FigureCard:
    return build_figure_card(
        FIGURE, "Online Time", metadata, session_run_id=SESSION_RUN_ID, notation=notation
    )


def test_system_scope_round_trips_and_extends_its_recipe_text(
    scope: RecipeScope, notation: NotationPolicy
) -> None:
    system = SystemScope(scope.dataset, scope.recipe, MatrixId(3), 7)
    plain = replace(notation, target=RenderTarget.PLAIN)

    assert decode_figure_scope(encode_figure_scope(system)) == system
    assert render_figure_scope(system, CARD_PROFILE, plain).text == (
        f"{render_figure_scope(scope, CARD_PROFILE, plain).text}, matrix 3, RHS 7"
    )


@pytest.fixture
def every_scope(scope: RecipeScope, figure_method: FigureMethod) -> dict[str, FigureScope]:
    return {
        "recipe": scope,
        "dataset": DatasetScope(scope.dataset),
        "all_datasets": AllDatasetsScope(),
        "method": MethodScope(figure_method.presentation),
        "system": SystemScope(scope.dataset, scope.recipe, MatrixId(3), 7),
    }


@pytest.mark.parametrize("kind", ["recipe", "dataset", "all_datasets", "method", "system"])
def test_card_of_each_scope_kind_round_trips_through_toml(
    kind: str,
    tmp_path: Path,
    every_scope: dict[str, FigureScope],
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    figure_method: FigureMethod,
    notation: NotationPolicy,
) -> None:
    card = _card(metadata_of(every_scope[kind]), notation)
    payload = card_payload(card)
    path = tmp_path / card_name(FIGURE)

    write_figure_card(path, card)

    assert cast(dict[str, Any], payload["scope"])["kind"] == kind
    assert [method["method_id"] for method in payload["methods"]] == ["jacobi"]
    assert payload["parameters"] == [
        {
            "rtol": 1e-6,
            "atol": 1e-14,
            "max_iterations": 100,
            "stopping_criterion": "residual_norm",
            "m_max": 5,
            "reference_precision_margin": 2.0,
            "timing_repeats": 3,
            "timing_warmup": 1,
            "timing_statistic": "median",
            "normalization": "matrix",
        }
    ]
    assert payload["source"] == {
        "session_run_id": SESSION_RUN_ID,
        "methods": [
            {
                "run_id": "method-run",
                "artifact_paths": [
                    "method_workload/setup_events.parquet",
                    "method_workload/summary.toml",
                ],
            }
        ],
    }
    assert read_figure_card(path) == card


def test_method_card_round_trips_without_group_only_fields(
    tmp_path: Path,
    scope: RecipeScope,
    method_metadata_of: Callable[[FigureScope], MethodFigureMetadata],
    notation: NotationPolicy,
) -> None:
    card = _card(method_metadata_of(scope), notation)
    path = tmp_path / card_name(FIGURE)

    write_figure_card(path, card)

    payload = card_payload(card)
    assert "failure_treatment" not in payload
    assert "scenario" not in payload
    assert read_figure_card(path) == card


def test_group_metadata_rejects_a_method_figure_kind(
    metadata_of: Callable[[FigureScope], GroupFigureMetadata], scope: RecipeScope
) -> None:
    with pytest.raises(ValueError, match="not a group figure kind"):
        replace(metadata_of(scope), kind=FigureKind.RESIDUAL_HISTORY)


def test_method_metadata_rejects_a_group_figure_kind(
    method_metadata_of: Callable[[FigureScope], MethodFigureMetadata], scope: RecipeScope
) -> None:
    with pytest.raises(ValueError, match="not a method figure kind"):
        replace(method_metadata_of(scope), kind=FigureKind.AGGREGATE_BARS)


def test_card_is_a_deterministic_function_of_its_value(
    tmp_path: Path,
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    notation: NotationPolicy,
) -> None:
    card = _card(metadata_of(scope), notation)
    path = tmp_path / card_name(FIGURE)
    write_figure_card(path, card)
    first = path.read_bytes()
    write_figure_card(path, card)

    assert path.read_bytes() == first


def test_card_stores_structured_scope_and_its_rendered_text(
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    notation: NotationPolicy,
) -> None:
    stored = cast(dict[str, Any], card_payload(_card(metadata_of(scope), notation))["scope"])

    assert decode_figure_scope(stored) == scope
    assert stored["kind"] == "recipe"
    assert (
        stored["display"]
        == "Test problem | Gaussian CG Residuals (k=20, N=100) - Sinusoidal (ω=6.28319)"
    )


@pytest.mark.parametrize(
    "reference",
    [
        NoReferenceConfigured(),
        ConfiguredReference(PreconditionerName("jacobi")),
        ReferenceUnavailable(PreconditionerName("ic0"), "setup skipped"),
    ],
)
def test_every_reference_selection_round_trips(
    reference: NoReferenceConfigured | ConfiguredReference | ReferenceUnavailable,
    tmp_path: Path,
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    notation: NotationPolicy,
) -> None:
    card = _card(replace(metadata_of(scope), reference_selection=reference), notation)
    path = tmp_path / card_name(FIGURE)
    write_figure_card(path, card)

    assert read_figure_card(path).metadata.reference_selection == reference


@pytest.mark.parametrize("name", list(ReportLevelName))
def test_every_report_level_name_round_trips_through_the_card(
    name: ReportLevelName,
    tmp_path: Path,
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    notation: NotationPolicy,
) -> None:
    card = _card(replace(metadata_of(scope), report_level=name), notation)
    path = tmp_path / card_name(FIGURE)
    write_figure_card(path, card)

    assert read_figure_card(path).metadata.report_level is name


@pytest.mark.parametrize("missing", ["parameters", "source", "methods", "counts", "reference"])
def test_card_missing_a_required_field_is_rejected(
    missing: str,
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    notation: NotationPolicy,
) -> None:
    payload = card_payload(_card(metadata_of(scope), notation))
    del payload[missing]

    with pytest.raises(InvalidFigureCardError, match=missing):
        validate_figure_card(payload)


def test_card_scope_without_display_text_is_rejected(
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    notation: NotationPolicy,
) -> None:
    payload = card_payload(_card(metadata_of(scope), notation))
    del payload["scope"]["display"]

    with pytest.raises(InvalidFigureCardError, match="display"):
        validate_figure_card(payload)


def test_card_source_without_run_id_is_rejected(
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
    notation: NotationPolicy,
) -> None:
    payload = card_payload(_card(metadata_of(scope), notation))
    del payload["source"]["methods"][0]["run_id"]

    with pytest.raises(InvalidFigureCardError, match="run_id"):
        validate_figure_card(payload)


@pytest.mark.parametrize("figure", ["", "nested/figure.png", "figure.svg"])
def test_card_must_name_a_png_by_its_basename(
    figure: str,
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
) -> None:
    with pytest.raises(ValueError):
        FigureCard(figure, "Title", "display", SESSION_RUN_ID, metadata_of(scope))


def test_staged_figure_file_must_carry_the_planned_name(
    tmp_path: Path,
    scope: RecipeScope,
    metadata_of: Callable[[FigureScope], GroupFigureMetadata],
) -> None:
    plan = FigurePlan("planned.png", "Title", metadata_of(scope))

    with pytest.raises(ValueError, match="planned name"):
        StagedFigure(plan, tmp_path / "other.png")


def test_collision_keys_equate_filesystem_unsafe_spellings() -> None:
    assert collision_key("a b.png") == collision_key("a_b.png")
    assert collision_key("A.png") == collision_key("a.png")


def test_slugs_follow_rendered_text_and_disambiguate_in_encounter_order() -> None:
    slugs = FigureSlugs()

    assert slugs.slug("method:a", "POD-2G (r=10)") == "POD-2G_r_10"
    assert slugs.slug("method:b", "POD-2G (r=10)") == "POD-2G_r_10_2"
    assert slugs.slug("method:c", "pod-2g (r=10)") == "pod-2g_r_10_3"
    assert slugs.slug("method:a", "ignored once allocated") == "POD-2G_r_10"


def test_slugs_are_truncated_and_stay_unique() -> None:
    slugs = FigureSlugs()
    long_text = "word " * 40

    first = slugs.slug("a", long_text)
    second = slugs.slug("b", long_text)

    assert len(first) <= MAX_SLUG_LENGTH
    assert len(second) <= MAX_SLUG_LENGTH
    assert first != second
