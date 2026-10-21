"""Real-batch acceptance coverage for setup-aware reporting."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.comparison_batch import run_comparison_batch
from neuralls.composition.comparison.models import ComparisonParams
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    METHOD_ARTIFACT_DIR,
)
from tests.workflows.comparison_support import make_settings, write_dataset_config


def _artifact_names(client: MlflowClient, run_id: str, path: str) -> set[str]:
    return {Path(item.path).name for item in client.list_artifacts(run_id, path=path)}


def _write_case(
    path: Path,
    *,
    tracking_uri: str,
    experiment: str,
    max_iterations: int = 20,
) -> None:
    payload = {
        "mlflow": {"tracking_uri": tracking_uri},
        "names": {"comparison": experiment},
        "comparison_baseline": {
            "rtol": 1e-6,
            "atol": 1e-14,
            "max_iterations": max_iterations,
            "stopping_criterion": "residual_norm",
            "m_max": 5,
            "normalize_system": "none",
            "reference_preconditioner": "jacobi",
            "preconditioners": [
                {"name": "identity", "type": "identity"},
                {"name": "jacobi", "type": "jacobi"},
            ],
        },
        "datasets": [{"id": "solutions", "path": "datasets/solutions.toml"}],
        "comparison_sweeps": [
            {
                "id_prefix": "task12",
                "matrix_dataset": "solutions",
                "rhs_source": {"kind": "gaussian"},
                "rhs_count": 1,
                "seed": 42,
                "selector": {"kind": "first_n", "n": 2},
            }
        ],
    }
    import tomli_w

    path.write_bytes(tomli_w.dumps(payload).encode("utf-8"))


@pytest.mark.integration
@pytest.mark.real_render
@pytest.mark.usefixtures("stub_input_gate")
def test_task12_real_batch_reconciles_topology_artifacts_and_reuse(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    sparse_id_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
) -> None:
    """Run the public batch twice, then invalidate it through solver settings."""
    settings = make_settings(tmp_path)
    dataset_dir = settings.processed_dir / "solutions"
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    sparse_id_dataset.rename(dataset_dir)
    pin_test_fold(dataset_dir, (0, 1, 2, 3))
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "solutions.toml", "solutions")

    case_path = tmp_path / "task12.toml"
    _write_case(case_path, tracking_uri=mlflow_tracking_uri, experiment=mlflow_experiment)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    first = run_comparison_batch(case_path, ComparisonParams(), settings)
    assert len(first) == 2 and all(outcome.success for outcome in first)
    experiments = client.get_experiment_by_name(mlflow_experiment)
    assert experiments is not None
    runs = client.search_runs([experiments.experiment_id], order_by=["attributes.start_time ASC"])
    roots = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    assert len(roots) == 1
    session = roots[0]
    children = [
        run for run in runs if run.data.tags.get("mlflow.parentRunId") == session.info.run_id
    ]
    assert {run.data.tags.get("phase") for run in children} == {"comparison_method"}
    assert {run.data.params["method"] for run in children} == {"identity", "jacobi"}
    assert not any(
        run.data.tags.get("mlflow.parentRunId") in {child.info.run_id for child in children}
        for run in runs
    )

    import polars as pl

    for child in children:
        assert _artifact_names(client, child.info.run_id, METHOD_ARTIFACT_DIR) >= {
            "setup_events.parquet",
            "solve_attempts.parquet",
            "summary.toml",
            "method_config.toml",
            "method_diagnostics.json",
        }
        setup_path = client.download_artifacts(
            child.info.run_id, f"{METHOD_ARTIFACT_DIR}/setup_events.parquet", str(tmp_path)
        )
        solve_path = client.download_artifacts(
            child.info.run_id, f"{METHOD_ARTIFACT_DIR}/solve_attempts.parquet", str(tmp_path)
        )
        assert pl.read_parquet(setup_path).height == 2
        assert pl.read_parquet(solve_path).height == 2

    report_names = _artifact_names(client, session.info.run_id, GROUP_REPORT_ARTIFACT_DIR)
    manifest_path = client.download_artifacts(
        session.info.run_id,
        f"{GROUP_REPORT_ARTIFACT_DIR}/figure_manifest.toml",
        str(tmp_path),
    )
    manifest = tomllib.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_names = set(manifest["figures"])
    assert report_names == manifest_names | {"figure_manifest.toml"}
    assert manifest_names
    for metadata in manifest["figures"].values():
        assert metadata["datasets"] in ([], ["solutions"])
        assert metadata["recipes"]
        assert metadata["recipe_weights"]
        assert metadata["n_matrices"] == 2
        assert metadata["n_planned_rhs"] == 2
        assert metadata["reference_selection"] == "configured:jacobi"
        assert all("comparison_method[method=" in source for source in metadata["sources"])

    second = run_comparison_batch(case_path, ComparisonParams(), settings)
    assert len(second) == 2 and all(outcome.success for outcome in second)
    assert (
        len(
            client.search_runs(
                [experiments.experiment_id], filter_string="tags.phase = 'session_comparison'"
            )
        )
        == 1
    )

    _write_case(
        case_path,
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        max_iterations=21,
    )
    invalidated = run_comparison_batch(case_path, ComparisonParams(), settings)
    assert len(invalidated) == 2 and all(outcome.success for outcome in invalidated)
    assert (
        len(
            client.search_runs(
                [experiments.experiment_id], filter_string="tags.phase = 'session_comparison'"
            )
        )
        == 2
    )
