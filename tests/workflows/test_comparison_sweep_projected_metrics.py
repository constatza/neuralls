"""End-to-end test: a real `comparison_sweeps` run, then its report, logs group metrics.

Unlike the preparation tests in `test_comparison_workflow.py`, this mocks no
solver code — it lets the real build/setup/solve pipeline run
(a trivial `Identity` preconditioner against a tiny 2x2 matrix, so it stays
fast) so the real method-major execution and the report workflow's
`report_all_levels` wiring are actually exercised, not just its
individually-unit-tested pieces in isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import tomli_w
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.comparison_batch import (
    run_comparison_batch,
)
from neuralls.composition.assignments.comparison_report import run_comparison_report
from neuralls.composition.comparison.models import ComparisonCompleted, ComparisonParams
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.tracking.report_sink import GROUP_REPORT_ARTIFACT_DIR
from tests.workflows.comparison_support import (
    make_settings,
    write_dataset_config,
)

pytestmark = pytest.mark.deferred_io("generation-io")

_LEVELS = ("all_matrix_sets", "matrix_set", "matrix_set_rhs_type")
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


@pytest.mark.usefixtures("stub_input_gate")
@pytest.mark.integration
@pytest.mark.figure_placeholders
def test_real_sweep_execution_logs_group_metrics_and_reports(
    tmp_path: Path,
    sparse_id_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
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
    assert all(isinstance(outcome, ComparisonCompleted) for outcome in outcomes)
    report = run_comparison_report(experiments_config, settings=settings)
    client = MlflowClient(tracking_uri=build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db"))
    experiment_id = client.get_run(report.report_run_id).info.experiment_id

    def run_of_level(level: str) -> Run:
        found = client.search_runs(
            experiment_ids=[experiment_id], filter_string=f"tags.report_level = '{level}'"
        )
        assert len(found) == 1
        return found[0]

    # One dataset and one recipe: every report level sees the same four planned,
    # attempted, and converged RHS, each in the metrics of its own run.
    for level in _LEVELS:
        metrics = run_of_level(level).data.metrics
        assert metrics["n_planned_rhs/identity"] == _N_PLANNED_RHS
        assert metrics["n_attempted_rhs/identity"] == _N_PLANNED_RHS
        assert metrics["n_converged/identity"] == _N_PLANNED_RHS
        assert metrics["success_rate/identity"] == 1
    assert (
        len(
            client.search_runs(
                experiment_ids=[experiment_id], filter_string="tags.report_level = 'system'"
            )
        )
        == _N_PLANNED_RHS
    )

    def artifact_names(run_id: str) -> set[str]:
        return {
            Path(item.path).name
            for item in client.list_artifacts(run_id, path=GROUP_REPORT_ARTIFACT_DIR)
        }

    def with_cards(figures: set[str]) -> set[str]:
        """Each group figure is stored beside the card and plotted table of the same stem."""
        return (
            figures
            | {name.replace(".png", ".toml") for name in figures}
            | {name.replace(".png", ".parquet") for name in figures}
        )

    recipe_stem = "Test_problem_Ones_Solutions_N_4_-_Dataset_RHS"
    dataset_stem = "Test_problem_Ones_Solutions_N_4"
    all_stem = "All_datasets"
    assert artifact_names(run_of_level("matrix_set_rhs_type").info.run_id) == with_cards(
        {
            f"aggregate_bars_{recipe_stem}.png",
            f"solve_time_bars_{recipe_stem}.png",
            f"online_time_bars_{recipe_stem}.png",
            f"performance_profile_solve_{recipe_stem}.png",
            f"performance_profile_online_{recipe_stem}.png",
            f"ecdf_iterations_{recipe_stem}.png",
            f"ecdf_time_solve_{recipe_stem}.png",
            f"ecdf_time_online_{recipe_stem}.png",
        }
    )
    assert artifact_names(run_of_level("matrix_set").info.run_id) == with_cards(
        {
            f"aggregate_bars_{dataset_stem}.png",
            f"solve_time_bars_{dataset_stem}.png",
            f"online_time_bars_{dataset_stem}.png",
        }
    )
    assert artifact_names(report.report_run_id) == with_cards(
        {
            f"aggregate_bars_{all_stem}.png",
            f"solve_time_bars_{all_stem}.png",
            f"online_time_bars_{all_stem}.png",
        }
    )
