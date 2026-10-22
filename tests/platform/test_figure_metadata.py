from pathlib import Path

import pytest

from neuralls.domain.solver.models.presentation import SolverTimingContext
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureManifest,
    FigureMetadata,
    FigurePlan,
    ReferenceSelectionKind,
    TimingScenario,
    write_figure_manifest,
)


def _metadata() -> FigureMetadata:
    return FigureMetadata(
        kind=FigureKind.PERFORMANCE_PROFILE,
        metric_definition="amortized online seconds",
        scenario=TimingScenario.ONLINE,
        report_level="fine",
        scope="Dataset X - sinusoidal omega=6.28319",
        datasets=("Dataset X",),
        recipes=("sinusoidal omega=6.28319",),
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


def test_manifest_round_trip_is_deterministic(tmp_path: Path) -> None:
    manifest = FigureManifest()
    metadata = _metadata()
    manifest.add("performance_profile_fine_x_online.png", metadata)
    path = tmp_path / "figure_manifest.toml"
    write_figure_manifest(path, manifest)
    first = path.read_text()
    write_figure_manifest(path, manifest)
    assert path.read_text() == first
    assert "setup_events.parquet" in first
    assert "performance_profile_fine_x_online.png" in first


def test_manifest_serializes_literal_population_and_solver_context() -> None:
    manifest = FigureManifest()
    manifest.add("performance_profile_fine_x_online.png", _metadata())

    assert manifest.as_dict() == {
        "schema": "figure-manifest-v1",
        "figures": {
            "performance_profile_fine_x_online.png": {
                "kind": "performance_profile",
                "metric_definition": "amortized online seconds",
                "report_level": "fine",
                "scope": "Dataset X - sinusoidal omega=6.28319",
                "datasets": ["Dataset X"],
                "recipes": ["sinusoidal omega=6.28319"],
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


def test_manifest_rejects_duplicate_and_sanitized_collisions() -> None:
    manifest = FigureManifest()
    metadata = _metadata()
    manifest.add("a b.png", metadata)
    with pytest.raises(ValueError):
        manifest.add("a_b.png", metadata)
    with pytest.raises(ValueError):
        manifest.add("a b.png", metadata)


def test_manifest_validates_plans_before_upload() -> None:
    manifest = FigureManifest()
    metadata = _metadata()
    manifest.add_plan(FigurePlan("first.png", "Title", "Scope", metadata))
    with pytest.raises(ValueError, match="colliding"):
        manifest.validate_plans((FigurePlan("first.png", "Title", "Scope", metadata),))
