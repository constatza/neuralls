"""End-to-end test: a real `comparison_sweeps` run logs a scaling-law projection.

Unlike the sibling tests in `test_comparison_workflow.py`, this does NOT mock
`compare_preconditioners` — it lets the real build/setup/solve pipeline run
(a trivial `Identity` preconditioner against a tiny 2x2 matrix, so it stays
fast) so the real `DatasetPreconditionerCache` + `report_all_levels`
wiring in `comparison_batch.py` is actually exercised, not just its
individually-unit-tested pieces in isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tomli_w

from neuralls.composition.assignments.comparison_batch import run_comparison_batch
from neuralls.composition.comparison.models import ComparisonParams
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from tests.workflows.comparison_support import (
    make_settings,
    write_dataset_config,
)

_GROUP_IDS = ("fine_solutions_dataset", "medium_solutions", "global")
_N_SYSTEMS = 4
"""2 held-out matrices x `rhs_count` 2."""


def _write_sweep_config(path: Path, tmp_path: Path) -> None:
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
                "rhs_source": {"kind": "dataset"},
                "rhs_count": 2,
                "selector": {"kind": "fraction", "p": 1.0, "seed": 0},
                "id_prefix": "sweep",
            }
        ],
    }
    (tmp_path / "datasets").mkdir(exist_ok=True)
    write_dataset_config(tmp_path / "datasets" / "solutions.toml", "solutions")
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)


@pytest.mark.usefixtures("stub_input_gate", "stub_log_comparison_artifacts", "stub_setup_tracking")
@pytest.mark.integration
def test_real_sweep_execution_logs_projected_metrics_and_scaling_plot(
    tmp_path: Path,
    sparse_id_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
    fake_mlflow: MagicMock,
) -> None:
    settings = make_settings(tmp_path)
    dataset_dir = settings.processed_dir / "solutions"
    dataset_dir.parent.mkdir(parents=True)
    sparse_id_dataset.rename(dataset_dir)
    pin_test_fold(dataset_dir, (0, 1, 2, 3))
    experiments_config = tmp_path / "experiments.toml"
    _write_sweep_config(experiments_config, tmp_path)

    outcomes = run_comparison_batch(experiments_config, ComparisonParams(), settings)

    # Two held-out matrices with two held-out rows each: 2 x rhs_count=2 systems.
    assert len(outcomes) == 4
    assert all(outcome.success for outcome in outcomes)
    assert all(outcome.payload is not None for outcome in outcomes)

    # One dataset and one recipe: the fine, medium and global levels each pool
    # the same 2 matrices x 2 RHS = 4 systems, reported under three distinct ids
    # (sanitized for MLflow: "/" -> "_").
    metric_calls = {call.args[0]: call.args[1] for call in fake_mlflow.log_metric.call_args_list}
    for group_id in _GROUP_IDS:
        assert metric_calls[f"n_systems/{group_id}/identity"] == _N_SYSTEMS
        assert f"avg_setup_cost_s/{group_id}/identity" in metric_calls
        assert f"avg_solve_time_per_system_s/{group_id}/identity" in metric_calls

    param_calls = {call.args[0]: call.args[1] for call in fake_mlflow.log_param.call_args_list}
    # Identity is a torchalg StaticSetupPreconditioner: setup is shareable
    # regardless of matrix identity, and this sweep has no fit/train step at all.
    for group_id in _GROUP_IDS:
        assert param_calls[f"share_setup/{group_id}/identity"] == "True"

    artifact_calls = fake_mlflow.log_artifact.call_args_list
    scaling_plot_calls = [
        call
        for call in artifact_calls
        if call.kwargs.get("artifact_path") == "projected_scaling_law"
    ]
    # The full per-group artifact set is pinned in tests/platform/test_comparison_sink.py; here
    # it is enough that every reported level uploads its scaling-law plot and numbers.
    uploaded = {Path(call.args[0]).name for call in scaling_plot_calls}
    assert uploaded >= {
        f"projected_scaling_law_{group_id}{suffix}"
        for group_id in _GROUP_IDS
        for suffix in (".png", ".toml")
    }
