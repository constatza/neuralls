"""End-to-end test: a real `comparison_sweeps` run logs group reports.

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
from neuralls.platform.tracking.comparison_tracking import GROUP_REPORT_ARTIFACT_DIR
from tests.workflows.comparison_support import (
    make_settings,
    write_dataset_config,
)

_GROUP_IDS = ("fine_solutions_dataset", "medium_solutions", "global")
_N_PLANNED_RHS = 4
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
def test_real_sweep_execution_logs_group_metrics_and_reports(
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

    # Two held-out matrices with two held-out rows each: 2 x rhs_count=2 RHS.
    assert len(outcomes) == 4
    assert all(outcome.success for outcome in outcomes)
    assert all(outcome.payload is not None for outcome in outcomes)

    # One dataset and one recipe: every report level sees the same four planned,
    # attempted, and converged RHS.
    metric_calls = {call.args[0]: call.args[1] for call in fake_mlflow.log_metric.call_args_list}
    for group_id in _GROUP_IDS:
        assert metric_calls[f"n_planned_rhs/{group_id}/identity"] == _N_PLANNED_RHS
        assert metric_calls[f"n_attempted_rhs/{group_id}/identity"] == _N_PLANNED_RHS
        assert metric_calls[f"n_converged/{group_id}/identity"] == _N_PLANNED_RHS
        assert metric_calls[f"success_rate/{group_id}/identity"] == 1

    group_report_calls = [
        call
        for call in fake_mlflow.log_artifact.call_args_list
        if call.kwargs.get("artifact_path") == GROUP_REPORT_ARTIFACT_DIR
    ]
    uploaded_names = [Path(call.args[0]).name for call in group_report_calls]
    uploaded = set(uploaded_names)
    assert uploaded == {
        "figure_manifest.toml",
        "aggregate_bars_fine_solutions_dataset.png",
        "solve_time_bars_fine_solutions_dataset.png",
        "online_time_bars_fine_solutions_dataset.png",
        "performance_profile_solve_fine_solutions_dataset.png",
        "performance_profile_online_fine_solutions_dataset.png",
        "ecdf_iterations_fine_solutions_dataset.png",
        "ecdf_time_solve_fine_solutions_dataset.png",
        "ecdf_time_online_fine_solutions_dataset.png",
        "aggregate_bars_medium_solutions.png",
        "solve_time_bars_medium_solutions.png",
        "online_time_bars_medium_solutions.png",
        "aggregate_bars_global.png",
        "solve_time_bars_global.png",
        "online_time_bars_global.png",
    }
    assert uploaded_names.count("figure_manifest.toml") == 1
