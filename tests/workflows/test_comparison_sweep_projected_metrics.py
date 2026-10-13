"""End-to-end test: a real `comparison_sweeps` run logs a scaling-law projection.

Unlike the sibling tests in `test_comparison_workflow.py`, this does NOT mock
`compare_preconditioners` — it lets the real build/setup/solve pipeline run
(a trivial `Identity` preconditioner against a tiny 2x2 matrix, so it stays
fast) so the real `SweepPreconditionerCache` + `aggregate_projected_results`
wiring in `comparison_batch.py` is actually exercised, not just its
individually-unit-tested pieces in isolation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import tomli_w

from neuralls.composition.assignments.comparison_batch import run_comparison_batch
from neuralls.composition.comparison.models import ComparisonParams
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from tests.workflows.test_comparison_workflow import (
    _build_sweep_dataset,
    _configure_mock_mlflow,
    _make_settings,
    _write_dataset_config,
)

_MLFLOW_MODULE = "neuralls.composition.assignments.comparison_batch.mlflow"
_COMPARISON_TRACKING_MLFLOW_MODULE = "neuralls.platform.tracking.comparison_tracking.mlflow"
_SETUP_TRACKING = "neuralls.composition.assignments.comparison_batch.setup_comparison_tracking"
_LOG_COMPARISON_ARTIFACTS = (
    "neuralls.composition.assignments.comparison_batch.log_comparison_artifacts_to_mlflow"
)


def _write_sweep_config(path: Path, dataset_dir: Path, tmp_path: Path) -> None:
    payload: dict[str, object] = {
        "mlflow": {"tracking_uri": build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")},
        "names": {"training": "neuralls-training", "comparison": "Comparisons"},
        "comparison_baseline": {
            "rtol": 1.0e-6,
            "atol": 1.0e-14,
            "max_iterations": 10,
            "stopping_criterion": "residual_norm",
            "m_max": 20,
            "preconditioners": [{"name": "identity", "type": "identity"}],
        },
        "datasets": [{"id": "solutions", "path": "datasets/solutions.toml"}],
        "comparison_sweeps": [
            {
                "matrix_dataset": "solutions",
                "rhs_source": {"kind": "dataset", "path": str(dataset_dir)},
                "selector": {"kind": "all"},
                "id_prefix": "sweep",
            }
        ],
    }
    (tmp_path / "datasets").mkdir(exist_ok=True)
    _write_dataset_config(tmp_path / "datasets" / "solutions.toml", "solutions")
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)


def test_real_sweep_execution_logs_projected_metrics_and_scaling_plot(
    tmp_path: Path,
) -> None:
    settings = _make_settings(tmp_path)
    dataset_dir = _build_sweep_dataset(settings)
    experiments_config = tmp_path / "experiments.toml"
    _write_sweep_config(experiments_config, dataset_dir, tmp_path)

    with (
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = run_comparison_batch(experiments_config, ComparisonParams(), settings)

    # _build_sweep_dataset's test fold has 4 rows; selector "all" sweeps every one.
    assert len(outcomes) == 4
    assert all(outcome.success for outcome in outcomes)
    assert all(outcome.payload is not None for outcome in outcomes)

    metric_calls = {call.args[0]: call.args[1] for call in mock_mlflow.log_metric.call_args_list}
    assert metric_calls["n_systems/sweep/identity"] == 4
    assert "avg_setup_cost_s/sweep/identity" in metric_calls
    assert "avg_solve_time_per_system_s/sweep/identity" in metric_calls

    param_calls = {call.args[0]: call.args[1] for call in mock_mlflow.log_param.call_args_list}
    # Identity is a torchalg StaticSetupPreconditioner: setup is shareable
    # regardless of same_matrix, and this sweep has no fit/train step at all.
    assert param_calls["share_setup/sweep/identity"] == "True"

    artifact_calls = mock_mlflow.log_artifact.call_args_list
    scaling_plot_calls = [
        call
        for call in artifact_calls
        if call.kwargs.get("artifact_path") == "projected_scaling_law"
    ]
    assert len(scaling_plot_calls) == 1
    assert Path(scaling_plot_calls[0].args[0]).suffix == ".png"
