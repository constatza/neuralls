from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from neuralls.domain.benchmarking.selection import ConfiguredReference
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.figure_scope import (
    FigureScope,
    RecipeScope,
    render_figure_scope,
)
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.figure_metadata import (
    MANIFEST_PROFILE,
    MAX_SLUG_LENGTH,
    FailureTreatment,
    FigureKind,
    FigureManifest,
    FigureMetadata,
    FigurePlan,
    FigureSlugs,
    InvalidFigureManifestError,
    ReferenceSelectionKind,
    TimingScenario,
    validate_figure_manifest,
    write_figure_manifest,
)
from neuralls.platform.storage.facts_codec import decode_figure_scope
from neuralls.shared.rhs_recipe import SinusoidalRecipe, recipe_key
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import PreconditionerName


def _metadata(scope: FigureScope) -> FigureMetadata:
    return FigureMetadata(
        kind=FigureKind.PERFORMANCE_PROFILE,
        metric_definition="amortized online seconds",
        scenario=TimingScenario.ONLINE,
        report_level=ReportLevelName.MATRIX_SET_RHS_TYPE,
        scope=scope,
        dataset_ids=("Dataset X",),
        recipes=(SinusoidalRecipe(omega=6.28319),),
        n_matrices=1,
        n_planned_rhs=4,
        reference_selection=ReferenceSelectionKind.CONFIGURED,
        failure_treatment=FailureTreatment.CENSORED,
        timing_policy="planned setup share + solve",
        recipe_counts=(4,),
        aggregation_rule="one recipe, four planned RHS",
        solver_contexts=(
            SolverTimingContext(
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
            ),
        ),
        sources=("setup_events.parquet", "solve_attempts.parquet", "summary.toml"),
    )


def _manifest_of(metadata: FigureMetadata) -> FigureManifest:
    manifest = FigureManifest()
    manifest.add("performance_profile_fine_x_online.png", metadata)
    return manifest


def _entry(manifest: FigureManifest, notation: NotationPolicy) -> dict[str, Any]:
    figures = cast(dict[str, dict[str, Any]], manifest.as_dict(notation)["figures"])
    return figures["performance_profile_fine_x_online.png"]


@pytest.fixture
def scope(dataset_presentation: DatasetPresentation) -> RecipeScope:
    return RecipeScope(dataset_presentation, SinusoidalRecipe(omega=6.28319))


def test_manifest_round_trip_is_deterministic(
    tmp_path: Path, scope: RecipeScope, notation: NotationPolicy
) -> None:
    manifest = FigureManifest()
    metadata = _metadata(scope)
    manifest.add("performance_profile_fine_x_online.png", metadata)
    path = tmp_path / "figure_manifest.toml"
    write_figure_manifest(path, manifest, notation)
    first = path.read_text(encoding="utf-8")
    write_figure_manifest(path, manifest, notation)
    assert path.read_text(encoding="utf-8") == first
    assert "setup_events.parquet" in first
    assert "performance_profile_fine_x_online.png" in first


def test_manifest_stores_structured_scope_and_its_rendered_text(
    scope: RecipeScope, notation: NotationPolicy
) -> None:
    manifest = FigureManifest()
    manifest.add("performance_profile_fine_x_online.png", _metadata(scope))

    stored = _entry(manifest, notation)["scope"]

    assert decode_figure_scope(stored) == scope
    expected = render_figure_scope(
        scope, MANIFEST_PROFILE, replace(notation, target=RenderTarget.PLAIN)
    ).text
    assert stored["display"] == expected
    assert stored["kind"] == "recipe"
    assert (
        stored["display"]
        == "Test problem | Gaussian CG Residuals (k=20, N=100) - Sinusoidal (ω=6.28319)"
    )


def test_manifest_serializes_literal_population_and_solver_context(
    scope: RecipeScope, notation: NotationPolicy
) -> None:
    manifest = FigureManifest()
    manifest.add("performance_profile_fine_x_online.png", _metadata(scope))
    payload = manifest.as_dict(notation)
    figures = cast(dict[str, dict[str, Any]], payload["figures"])
    scope_table = figures["performance_profile_fine_x_online.png"].pop("scope")
    assert scope_table["kind"] == "recipe"

    assert payload == {
        "figures": {
            "performance_profile_fine_x_online.png": {
                "kind": "performance_profile",
                "metric_definition": "amortized online seconds",
                "report_level": "matrix_set_rhs_type",
                "dataset_ids": ["Dataset X"],
                "recipes": [recipe_key(SinusoidalRecipe(omega=6.28319))],
                "recipe_counts": [4],
                "n_matrices": 1,
                "n_planned_rhs": 4,
                "reference_selection": "configured",
                "failure_treatment": "censored",
                "timing_policy": "planned setup share + solve",
                "aggregation_rule": "one recipe, four planned RHS",
                "recipe_weights": [],
                "solver_contexts": [
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
                ],
                "sources": ["setup_events.parquet", "solve_attempts.parquet", "summary.toml"],
                "scenario": "online",
            }
        },
    }


def _manifest_payload(scope: RecipeScope, notation: NotationPolicy) -> dict[str, Any]:
    manifest = FigureManifest()
    manifest.add("performance_profile_fine_x_online.png", _metadata(scope))
    return manifest.as_dict(notation)


def test_written_manifest_passes_validation(scope: RecipeScope, notation: NotationPolicy) -> None:
    validate_figure_manifest(_manifest_payload(scope, notation))


def test_manifest_without_figures_table_is_rejected() -> None:
    with pytest.raises(InvalidFigureManifestError, match="figures"):
        validate_figure_manifest({})


def test_manifest_entry_missing_required_field_is_rejected(
    scope: RecipeScope, notation: NotationPolicy
) -> None:
    payload = _manifest_payload(scope, notation)
    del payload["figures"]["performance_profile_fine_x_online.png"]["solver_contexts"]
    with pytest.raises(InvalidFigureManifestError, match="solver_contexts"):
        validate_figure_manifest(payload)


def test_manifest_scope_without_display_text_is_rejected(
    scope: RecipeScope, notation: NotationPolicy
) -> None:
    payload = _manifest_payload(scope, notation)
    del payload["figures"]["performance_profile_fine_x_online.png"]["scope"]["display"]
    with pytest.raises(InvalidFigureManifestError, match="scope"):
        validate_figure_manifest(payload)


def test_manifest_rejects_duplicate_and_sanitized_collisions(scope: RecipeScope) -> None:
    manifest = FigureManifest()
    metadata = _metadata(scope)
    manifest.add("a b.png", metadata)
    with pytest.raises(ValueError):
        manifest.add("a_b.png", metadata)
    with pytest.raises(ValueError):
        manifest.add("a b.png", metadata)


def test_manifest_validates_plans_before_upload(scope: RecipeScope) -> None:
    manifest = FigureManifest()
    metadata = _metadata(scope)
    manifest.add_plan(FigurePlan("first.png", "Title", metadata))
    with pytest.raises(ValueError):
        manifest.validate_plans((FigurePlan("first.png", "Title", metadata),))


def test_manifest_keeps_machine_ids_in_id_fields_and_the_reference_as_a_kind(
    scope: RecipeScope, notation: NotationPolicy
) -> None:
    from dataclasses import replace

    manifest = FigureManifest()
    metadata = replace(
        _metadata(scope), reference_selection=ConfiguredReference(PreconditionerName("jacobi-7"))
    )
    manifest.add("performance_profile_fine_x_online.png", metadata)

    stored = _entry(manifest, notation)

    assert stored["dataset_ids"] == ["Dataset X"]
    assert stored["reference_selection"] == "configured"
    assert stored["reference_method_id"] == "jacobi-7"
    assert "datasets" not in stored
    assert stored["scope"] == _entry(_manifest_of(_metadata(scope)), notation)["scope"]


@pytest.mark.parametrize("name", list(ReportLevelName))
def test_every_report_level_name_round_trips_through_the_manifest(
    name: ReportLevelName, scope: RecipeScope, notation: NotationPolicy
) -> None:
    manifest = FigureManifest()
    manifest.add("figure.png", replace(_metadata(scope), report_level=name))
    payload = manifest.as_dict(notation)

    validate_figure_manifest(payload)
    figures = cast(dict[str, dict[str, Any]], payload["figures"])
    assert ReportLevelName(figures["figure.png"]["report_level"]) is name


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
