"""Lightweight MLflow artifact-location tests using local SQLite tracking."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pytest
import tomli_w
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from tests.dataset_support import build_test_dataset

from neuralls.composition.assignments import comparison_report
from neuralls.composition.assignments._training_artifacts import (
    _log_training_evaluation,
    _normalize_training_numpy_payload,
)
from neuralls.composition.assignments.comparison_batch import run_comparison_batch
from neuralls.composition.assignments.comparison_report import (
    _finished_report,
    run_comparison_report,
)
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.comparison.models import (
    ComparisonCompleted,
    ComparisonParams,
    ComparisonReused,
)
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.solver.models.figure_scope import SystemScope
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.reporting.benchmark_events import MethodBenchmarkArtifacts
from neuralls.platform.reporting.figure_card import read_figure_card
from neuralls.platform.reporting.figure_index import FIGURE_INDEX_FILE, read_figure_index
from neuralls.platform.reporting.figure_metadata import NoTable, SiblingTable
from neuralls.platform.reporting.figure_table import read_group_table
from neuralls.platform.reporting.training_diagnostics import compute_diagnostics
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    METHOD_FIGURE_ARTIFACT_DIR,
)
from neuralls.platform.tracking.mlflow_client import log_diagnostics_to_mlflow
from neuralls.platform.tracking.session_artifacts import REPORT_PHASE


def _artifact_files(client: MlflowClient, run_id: str, path: str | None = None) -> list[str]:
    """Every file artifact path of a run, recursing through directories."""
    files: list[str] = []
    for item in client.list_artifacts(run_id, path=path):
        if item.is_dir:
            files.extend(_artifact_files(client, run_id, item.path))
        else:
            files.append(item.path)
    return files


def _report_tree_run_ids(client: MlflowClient, root_id: str) -> list[str]:
    """The root run and every report run nested under it."""
    experiment_id = client.get_run(root_id).info.experiment_id
    found = [root_id]
    for run in client.search_runs(
        experiment_ids=[experiment_id], filter_string=f"tags.mlflow.parentRunId = '{root_id}'"
    ):
        found.extend(_report_tree_run_ids(client, run.info.run_id))
    return found


def _assert_report_tree_is_documented(
    client: MlflowClient,
    *,
    report_id: str,
    session_id: str,
    method_run_ids: set[str],
    download_root: Path,
) -> None:
    """Every figure of the tree has a same-stem card, an index entry and real provenance."""
    figures: set[tuple[str, str]] = set()
    for run_id in _report_tree_run_ids(client, report_id):
        files = _artifact_files(client, run_id)
        for png in (path for path in files if path.endswith(".png")):
            figures.add((run_id, png))
            card_path = png.removesuffix(".png") + ".toml"
            assert card_path in files
            local = client.download_artifacts(
                run_id, card_path, dst_path=str(download_root / run_id)
            )
            card = read_figure_card(Path(local))
            assert card.figure == Path(png).name
            match card.table:
                case SiblingTable(name=name):
                    table_path = str(Path(png).with_name(name))
                    assert table_path in files
                    table_local = client.download_artifacts(
                        run_id, table_path, dst_path=str(download_root / run_id)
                    )
                    assert read_group_table(Path(table_local)).get_column(
                        "method_id"
                    ).to_list() == ["identity"]
                case NoTable():
                    assert png.removesuffix(".png") + ".parquet" not in files
            assert card.session_run_id == session_id
            assert card.metadata.sources
            assert {source.run_id for source in card.metadata.sources} <= method_run_ids
            assert {method.method_id for method in card.metadata.methods} == {"identity"}
            assert len(card.metadata.solver_contexts) == 1
    assert figures

    index_local = client.download_artifacts(
        report_id, FIGURE_INDEX_FILE, dst_path=str(download_root / "index")
    )
    index = read_figure_index(Path(index_local))
    assert {(entry.run_id, entry.artifact_path) for entry in index} == figures
    for entry in index:
        assert entry.card in _artifact_files(client, entry.run_id)

    systems = [
        read_figure_card(
            Path(
                client.download_artifacts(entry.run_id, entry.card, str(download_root / "systems"))
            )
        )
        for entry in index
        if entry.level is ReportLevelName.SYSTEM
        and Path(entry.artifact_path).name == "residual_history.png"
    ]
    assert len(systems) == 2
    assert len({card.metadata.scope for card in systems}) == 2
    assert all(isinstance(card.metadata.scope, SystemScope) for card in systems)


def _sqlite_tracking_uri(db_path: Path) -> str:
    """Build sqlite tracking URI from DB path."""
    return build_sqlite_tracking_uri(db_path)


def _ensure_experiment(
    *,
    client: MlflowClient,
    name: str,
    artifact_root: Path,
) -> str:
    """Create experiment if needed and return id."""
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    try:
        return client.create_experiment(
            name=name,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    except MlflowException:
        existing = client.get_experiment_by_name(name)
        if existing is None:
            raise
        return existing.experiment_id


def test_training_logs_diagnostics_artifact_to_mlflow_with_sqlite(tmp_path: Path) -> None:
    """Training diagnostics are logged under figures/ in the active MLflow run."""
    output_root = tmp_path / "output"
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="training-diagnostics-sqlite",
        artifact_root=artifact_root,
    )

    mlflow.set_tracking_uri(tracking_uri)
    predictions = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    targets = np.array([[1.1], [1.9], [3.2]], dtype=np.float64)
    contract = default_training_dataset_contract()
    normalized_payload = _normalize_training_numpy_payload(
        {
            "predictions": {"output": predictions.flatten()},
            "targets": {"y": targets.flatten()},
        },
        contract,
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name="diag-train") as run:
        run_id = run.info.run_id

    _log_training_evaluation(
        tracking_uri=tracking_uri,
        run_id=run_id,
        numpy_payload=normalized_payload,
        metrics_dir=tmp_path / "metrics",
        contract=contract,
    )

    run_data = client.get_run(run_id).data
    assert "eval/mae" in run_data.metrics
    assert "eval/mse" in run_data.metrics
    assert "eval/rmse" in run_data.metrics


def test_log_diagnostics_to_mlflow_reopens_sqlite_run_without_active_context(
    tmp_path: Path,
) -> None:
    """Diagnostics helper logs directly to an existing SQLite-backed run."""
    output_root = tmp_path / "output"
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="training-diagnostics-direct-sqlite",
        artifact_root=artifact_root,
    )

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=experiment_id, run_name="diag-direct") as run:
        run_id = run.info.run_id

    diagnostics = compute_diagnostics(
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([[1.1], [1.9]], dtype=np.float64),
    )

    log_diagnostics_to_mlflow(tracking_uri, run_id, diagnostics)

    run_data = client.get_run(run_id).data
    assert "eval/mae" in run_data.metrics
    assert "eval/mse" in run_data.metrics


def _write_dataset_config(path: Path, dataset_id: str, data_dir: Path) -> None:
    """Write a minimal dataset config for the registry."""
    path.write_text(
        "\n".join(
            [
                f'id = "{dataset_id}"',
                "",
                "[output]",
                f'data_dir = "{data_dir.as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_experiments_config(
    path: Path,
    output_root: Path,
    *,
    dataset_dir: Path | None = None,
    method_config_path: Path | None = None,
) -> str:
    """Write sqlite-only experiments topology and return tracking URI.

    Args:
        path: Output path for the experiments TOML.
        output_root: Output root directory for MLflow tracking.
        dataset_dir: Optional dataset directory for registry dataset configs.
        method_config_path: Optional method override TOML path for [[comparisons]].

    Returns:
        SQLite tracking URI string.
    """
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    payload: dict[str, object] = {
        "mlflow": {
            "tracking_uri": tracking_uri,
        },
        "names": {
            "training": "neuralls-training",
            "comparison": "Comparisons",
        },
        "comparison_baseline": {
            "rtol": 1e-6,
            "atol": 1e-14,
            "max_iterations": 20,
            "stopping_criterion": "residual_norm",
            "m_max": 5,
            "normalize_system": "matrix",
            "preconditioners": [{"name": "identity", "type": "identity"}],
        },
    }
    if dataset_dir is not None:
        datasets_dir = path.parent / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        _write_dataset_config(datasets_dir / "benchmark.toml", "benchmark", dataset_dir.parent)
        payload["datasets"] = [{"id": "benchmark", "path": "datasets/benchmark.toml"}]
        comparison_entry: dict[str, object] = {
            "id": "benchmark-comparison",
            "matrix_dataset": "benchmark",
            "rhs_source": {"kind": "gaussian"},
            "rhs_count": 2,
        }
        if method_config_path is not None:
            comparison_entry["method"] = method_config_path.relative_to(path.parent).as_posix()
        payload["comparisons"] = [comparison_entry]
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)
    return tracking_uri


def _write_method_config(path: Path) -> None:
    """Write a minimal comparison method override config."""
    path.write_text(
        '[general]\n\n[general.params]\nrtol = 1e-6\natol = 1e-14\nmax_iterations = 20\nstopping_criterion = "residual_norm"\nm_max = 5\n\n[[preconditioners]]\nname = "identity"\ntype = "identity"\n',
        encoding="utf-8",
    )


@pytest.mark.usefixtures("stub_input_gate")
@pytest.mark.integration
@pytest.mark.figure_placeholders
def test_comparison_logs_artifacts_to_mlflow_with_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comparison workflow logs figures and diagnostics files under run artifacts."""
    output_root = tmp_path / "output"
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = tmp_path / "benchmark"
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, np.eye(2, dtype=np.float64))
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 1},
                seed=42,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="hdf5",
    )

    # Write a method config to test that config artifacts are logged under config/.
    method_config = tmp_path / "method.toml"
    _write_method_config(method_config)

    experiments_config = tmp_path / "experiments.toml"
    tracking_uri = _write_experiments_config(
        experiments_config,
        output_root,
        dataset_dir=dataset_dir,
        method_config_path=method_config,
    )

    # Ensure comparison experiment uses a deterministic local artifact root.
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="Comparisons",
        artifact_root=artifact_root,
    )
    legacy = client.create_run(experiment_id, tags={"phase": "comparison"})
    client.set_terminated(legacy.info.run_id, status="FINISHED")

    outcomes = run_comparison_batch(
        case_config_path=experiments_config,
        params=ComparisonParams(),
    )

    assert outcomes and isinstance(outcomes[0], ComparisonCompleted)

    # The session owns cross-method reports and has one direct method child.
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.phase = 'session_comparison'",
        max_results=1,
    )
    assert runs
    session_id = runs[0].info.run_id
    method_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{session_id}' and tags.phase = 'comparison_method'",
    )
    assert len(method_runs) == 1
    method_run = method_runs[0]
    assert not client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{method_run.info.run_id}'",
    )

    method_artifacts = client.list_artifacts(method_run.info.run_id, path="method_workload")
    method_names = {Path(item.path).name for item in method_artifacts}
    assert set(MethodBenchmarkArtifacts.file_names()).issubset(method_names)
    assert not [
        item
        for item in client.list_artifacts(method_run.info.run_id)
        if item.path != "method_workload"
    ]

    report = run_comparison_report(experiments_config)
    assert report.session_run_id == session_id
    assert not report.reused
    reports = MlflowClient(tracking_uri=tracking_uri)
    report_id = report.report_run_id
    method_figures = reports.list_artifacts(report_id, path=METHOD_FIGURE_ARTIFACT_DIR)
    assert {Path(item.path).name for item in method_figures} == {
        "residual_history_Method_Identity.png",
        "residual_history_Method_Identity.toml",
        "energy_history_Method_Identity.png",
        "energy_history_Method_Identity.toml",
    }

    def child(parent_id: str, level: str) -> str:
        found = client.search_runs(
            experiment_ids=[reports_experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_id}' and tags.report_level = '{level}'",
        )
        assert len(found) == 1
        return found[0].info.run_id

    def group_artifacts(run_id: str) -> list[str]:
        return [
            Path(item.path).name
            for item in client.list_artifacts(run_id, path=GROUP_REPORT_ARTIFACT_DIR)
        ]

    reports_experiment_id = client.get_run(report_id).info.experiment_id
    matrix_set_id = child(report_id, "matrix_set")
    cohort_id = child(matrix_set_id, "matrix_set_rhs_type")
    system_runs = client.search_runs(
        experiment_ids=[reports_experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{cohort_id}' and tags.report_level = 'system'",
    )
    assert len(system_runs) == 2
    root_names = group_artifacts(report_id)
    assert "figure_manifest.toml" not in root_names
    assert {
        "aggregate_bars_All_datasets.png",
        "solve_time_bars_All_datasets.png",
        "online_time_bars_All_datasets.png",
    } <= set(root_names)
    assert {
        "aggregate_bars_Test_problem_Ones_Solutions_N_1.png",
        "solve_time_bars_Test_problem_Ones_Solutions_N_1.png",
        "online_time_bars_Test_problem_Ones_Solutions_N_1.png",
    } <= set(group_artifacts(matrix_set_id))
    recipe_stem = "Test_problem_Ones_Solutions_N_1_-_Gaussian"
    assert {
        f"aggregate_bars_{recipe_stem}.png",
        f"performance_profile_solve_{recipe_stem}.png",
        f"performance_profile_online_{recipe_stem}.png",
        f"ecdf_iterations_{recipe_stem}.png",
        f"ecdf_time_solve_{recipe_stem}.png",
        f"ecdf_time_online_{recipe_stem}.png",
    } <= set(group_artifacts(cohort_id))
    for system in system_runs:
        assert {Path(item.path).name for item in client.list_artifacts(system.info.run_id)} == {
            "residual_history.png",
            "residual_history.toml",
            "energy_history.png",
            "energy_history.toml",
        }
        assert set(system.data.metrics) == {
            "iterations/identity",
            "solve_s/identity",
            "converged/identity",
        }

    matrix_set_params = client.get_run(matrix_set_id).data.params
    cohort_params = client.get_run(cohort_id).data.params
    assert set(matrix_set_params) == {"report_level", "dataset_id"}
    assert matrix_set_params["report_level"] == "matrix_set"
    assert cohort_params["report_level"] == "matrix_set_rhs_type"
    assert set(cohort_params) == {"report_level", "dataset_id", "recipe_key"}
    assert cohort_params["dataset_id"] == matrix_set_params["dataset_id"]
    system_params = [system.data.params for system in system_runs]
    assert all(
        set(params) == {*cohort_params, "matrix_id", "rhs_ordinal"}
        and params["report_level"] == "system"
        and params["dataset_id"] == cohort_params["dataset_id"]
        and params["recipe_key"] == cohort_params["recipe_key"]
        for params in system_params
    )
    assert len({(p["matrix_id"], p["rhs_ordinal"]) for p in system_params}) == 2
    assert client.get_run(report_id).data.params == {"report_level": "all_matrix_sets"}

    _assert_report_tree_is_documented(
        client,
        report_id=report_id,
        session_id=session_id,
        method_run_ids={method_run.info.run_id},
        download_root=tmp_path / "cards",
    )
    legacy_runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.phase = 'comparison'"
    )
    assert [run.info.run_id for run in legacy_runs] == [legacy.info.run_id]

    session_count = len(
        client.search_runs(
            experiment_ids=[experiment_id], filter_string="tags.phase = 'session_comparison'"
        )
    )
    reused = run_comparison_batch(case_config_path=experiments_config, params=ComparisonParams())
    assert reused and isinstance(reused[0], ComparisonReused)
    assert (
        len(
            client.search_runs(
                experiment_ids=[experiment_id], filter_string="tags.phase = 'session_comparison'"
            )
        )
        == session_count
    )

    assert run_comparison_report(experiments_config).reused

    forced = run_comparison_batch(
        case_config_path=experiments_config, params=ComparisonParams(force=True)
    )
    assert forced and isinstance(forced[0], ComparisonCompleted)
    assert (
        len(
            client.search_runs(
                experiment_ids=[experiment_id], filter_string="tags.phase = 'session_comparison'"
            )
        )
        == session_count + 1
    )

    # A failure while a report is written leaves its root run FAILED, so it is never reused.
    def fail_report(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("report boom")

    monkeypatch.setattr(comparison_report, "report_all_levels", fail_report)
    with pytest.raises(RuntimeError, match="report boom"):
        run_comparison_report(experiments_config)
    failed = client.search_runs(
        experiment_ids=[reports_experiment_id],
        filter_string=f"tags.phase = '{REPORT_PHASE}' and attributes.status = 'FAILED'",
    )
    assert len(failed) == 1
    assert failed[0].data.tags["source_session_run_id"] != session_id
    failed_tags = failed[0].data.tags
    source_tags = {
        key: failed_tags[key]
        for key in ("source_session_run_id", "notation_digest", "report_schema_version")
    }
    report_experiment = client.get_experiment(reports_experiment_id).name
    assert _finished_report(client, report_experiment, source_tags) is None
