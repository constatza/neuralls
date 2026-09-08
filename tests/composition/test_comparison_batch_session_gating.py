"""run_comparison_batch must never open an MLflow session parent run unless at
least one [[comparisons]] entry genuinely needs execution.

Complements test_comparison_workflow.py's
test_run_comparison_batch_preserves_declared_order (which also asserts the
all-cache-hit case never calls mlflow.start_run) with the mixed case: one
cache hit alongside one entry that needs real execution.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

from neuralls.composition.assignments.comparison_batch import (
    ResolvedComparisonSpecs,
    _ComparisonResolutionContext,
    _PreparedComparisonExecution,
    run_comparison_batch,
)
from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonParams
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from tests.workflows.test_comparison_workflow import _mock_cfg, _write_experiments_config

_MLFLOW_MODULE = "neuralls.composition.assignments.comparison_batch.mlflow"


def test_run_comparison_batch_opens_no_run_when_every_entry_is_a_cache_hit(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        entry: ComparisonRegistryEntry,
        context: object,
        *,
        force: bool = False,
    ) -> ComparisonOutcome:
        return ComparisonOutcome(
            comparison_id=entry.id,
            comparison_display_name=entry.effective_display_name,
            success=True,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=_mock_cfg(),
        ),
        patch(_MLFLOW_MODULE) as mock_mlflow,
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert [outcome.success for outcome in outcomes] == [True, True]
    mock_mlflow.start_run.assert_not_called()


def test_run_comparison_batch_opens_a_run_when_one_entry_needs_execution(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        entry: ComparisonRegistryEntry,
        context: _ComparisonResolutionContext,
        *,
        force: bool = False,
    ) -> ComparisonOutcome | _PreparedComparisonExecution:
        if entry.id == "a":
            return ComparisonOutcome(
                comparison_id=entry.id,
                comparison_display_name=entry.effective_display_name,
                success=True,
            )
        return _PreparedComparisonExecution(
            cfg=cfg,
            entry=entry,
            topology=context.topology,
            cleanup=contextlib.ExitStack(),
            resolved=ResolvedComparisonSpecs(
                specs=[], warnings=(), checkpoint_dependency_hash="hash-b"
            ),
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=_mock_cfg(),
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch._execute_prepared_comparison",
            return_value=ComparisonOutcome(
                comparison_id="b", comparison_display_name="b", success=True
            ),
        ),
        patch(_MLFLOW_MODULE) as mock_mlflow,
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert [outcome.comparison_id for outcome in outcomes] == ["a", "b"]
    mock_mlflow.start_run.assert_called_once()
