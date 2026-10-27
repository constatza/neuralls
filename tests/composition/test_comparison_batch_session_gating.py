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
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments.comparison_batch import (
    ResolvedComparisonSpecs,
    _ComparisonResolutionContext,
    _PreparedComparisonExecution,
    run_comparison_batch,
)
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    ComparisonParams,
    ComparisonReused,
    PlannedSystem,
)
from neuralls.domain.identity import StageIdentity
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description import PreparedComparisonPresentation
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonBatchPhase,
)
from tests.workflows.comparison_support import (
    MLFLOW_MODULE,
    mock_cfg,
    write_experiments_config,
)


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_run_comparison_batch_opens_no_run_when_every_entry_is_a_cache_hit(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        planned: PlannedSystem,
        context: object,
        *,
        force: bool = False,
    ) -> ComparisonOutcome:
        return ComparisonReused(
            comparison_id=planned.entry.id,
            comparison_display_name=planned.entry.effective_display_name,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=mock_cfg(),
        ),
        patch(MLFLOW_MODULE) as mock_mlflow,
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert all(isinstance(outcome, ComparisonReused) for outcome in outcomes)
    mock_mlflow.start_run.assert_not_called()


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_run_comparison_batch_opens_a_run_when_one_entry_needs_execution(
    tmp_path: Path,
    prepared_presentation: PreparedComparisonPresentation,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        planned: PlannedSystem,
        context: _ComparisonResolutionContext,
        *,
        force: bool = False,
    ) -> ComparisonOutcome | _PreparedComparisonExecution:
        if planned.entry.id == "a":
            return ComparisonReused(
                comparison_id=planned.entry.id,
                comparison_display_name=planned.entry.effective_display_name,
            )
        return _PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=contextlib.ExitStack(),
            resolved=ResolvedComparisonSpecs(specs=[], warnings=(), checkpoint_digests={}),
            identity=StageIdentity.build(
                "comparison", {"entry": canonical_digest(planned.entry.id)}
            ),
            settings=context.settings,
            presentation=prepared_presentation,
            title=prepared_presentation.scope,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=mock_cfg(),
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch._has_complete_reusable_session",
            return_value=False,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch._execute_all",
            return_value=(),
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch._outcomes_from_workloads",
            return_value=[
                ComparisonReused(comparison_id="a", comparison_display_name="a"),
                ComparisonReused(comparison_id="b", comparison_display_name="b"),
            ],
        ),
        patch("neuralls.composition.assignments.comparison_batch.report_all_levels"),
        patch("neuralls.composition.assignments.comparison_batch.MLflowGroupReportSink"),
        patch(MLFLOW_MODULE) as mock_mlflow,
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert [outcome.comparison_id for outcome in outcomes] == ["a", "b"]
    mock_mlflow.start_run.assert_called_once()


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_run_comparison_batch_records_and_reraises_blocking_execution_failure(
    tmp_path: Path,
    prepared_presentation: PreparedComparisonPresentation,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    write_experiments_config(experiments_config, with_comparisons=True)
    original = RuntimeError("injected method failure")
    failure_sink = MagicMock()

    def prepare(
        cfg: object,
        planned: PlannedSystem,
        context: _ComparisonResolutionContext,
        *,
        force: bool = False,
    ) -> _PreparedComparisonExecution:
        del force
        return _PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=contextlib.ExitStack(),
            resolved=ResolvedComparisonSpecs(specs=[], warnings=(), checkpoint_digests={}),
            identity=StageIdentity.build(
                "comparison", {"entry": canonical_digest(planned.entry.id)}
            ),
            settings=context.settings,
            presentation=prepared_presentation,
            title=prepared_presentation.scope,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=prepare,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=mock_cfg(),
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch._has_complete_reusable_session",
            return_value=False,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch._execute_all",
            side_effect=original,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.MLflowWorkflowFailureSink",
            return_value=failure_sink,
        ),
        patch(MLFLOW_MODULE) as mock_mlflow,
        pytest.raises(RuntimeError) as raised,
    ):
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        mock_mlflow.start_run.return_value.__exit__.return_value = False
        run_comparison_batch(experiments_config, ComparisonParams())

    assert raised.value is original
    context = failure_sink.record.call_args.args[0]
    assert context == ComparisonBatchFailureContext(
        phase=ComparisonBatchPhase.METHOD_WORKLOADS,
        case_config_path=experiments_config.resolve(),
    )
