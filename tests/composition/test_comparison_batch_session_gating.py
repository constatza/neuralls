"""run_comparison_batch must never open an MLflow session parent run unless at
least one [[comparisons]] entry genuinely needs execution.

Complements test_comparison_workflow.py's
test_run_comparison_batch_preserves_declared_order (which also asserts the
all-cache-hit case never calls mlflow.start_run) with the mixed case: one
cache hit alongside one entry that needs real execution.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments.comparison_batch import (
    run_loaded_comparison_batch,
)
from neuralls.composition.assignments.comparison_prepared import (
    PendingComparison,
    PreparedComparisonExecution,
)
from neuralls.composition.assignments.comparison_resolution import (
    ComparisonResolutionContext,
    ResolvedComparisonSpecs,
)
from neuralls.composition.comparison.method_data_origin import NoFitData
from neuralls.composition.comparison.models import (
    ComparisonParams,
    ComparisonReused,
    PlannedSystem,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import PreparedComparisonPresentation
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.tracking.services import MlflowTrackingServices
from neuralls.shared.digest import canonical_digest
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonBatchPhase,
)
from tests.workflows.comparison_support import (
    MLFLOW_MODULE,
    VIRTUAL_CASE_PATH,
    in_memory_case,
    mock_cfg,
    virtual_settings,
)


@pytest.mark.usefixtures("stub_input_gate", "stub_planning", "stub_setup_tracking")
def test_run_comparison_batch_opens_no_run_when_every_entry_is_a_cache_hit(
    comparison_run: NamedRun,
    fake_tracking: MlflowTrackingServices,
) -> None:
    case = in_memory_case(with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        planned: PlannedSystem,
        context: object,
        *,
        force: bool = False,
    ) -> ComparisonReused:
        return ComparisonReused(
            comparison_id=planned.entry.id,
            run=comparison_run,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_prepare.prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ),
        patch(
            "neuralls.composition.assignments.comparison_prepare.resolve_comparison_config",
            return_value=mock_cfg(),
        ),
        patch(MLFLOW_MODULE) as mock_mlflow,
    ):
        outcomes = run_loaded_comparison_batch(
            case,
            VIRTUAL_CASE_PATH,
            ComparisonParams(),
            virtual_settings(),
            tracking=fake_tracking,
        )

    assert all(isinstance(outcome, ComparisonReused) for outcome in outcomes)
    mock_mlflow.start_run.assert_not_called()


@pytest.mark.usefixtures("stub_input_gate", "stub_planning", "stub_setup_tracking")
def test_run_comparison_batch_opens_a_run_when_one_entry_needs_execution(
    prepared_presentation: PreparedComparisonPresentation,
    comparison_run: NamedRun,
    fake_tracking: MlflowTrackingServices,
) -> None:
    case = in_memory_case(with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        planned: PlannedSystem,
        context: ComparisonResolutionContext,
        *,
        force: bool = False,
    ) -> PendingComparison:
        if planned.entry.id == "a":
            return ComparisonReused(
                comparison_id=planned.entry.id,
                run=comparison_run,
            )
        return PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=contextlib.ExitStack(),
            resolved=ResolvedComparisonSpecs(
                specs=[StandardPreconditionerConfig(type=PreconditionerType.JACOBI)],
                warnings=(),
                checkpoint_digests={},
            ),
            origins_by_method={"jacobi": NoFitData()},
            training_run_costs={},
            identity=StageIdentity.build(
                "comparison", {"entry": canonical_digest(planned.entry.id)}
            ),
            settings=context.settings,
            presentation=prepared_presentation,
            title=prepared_presentation.scope,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_prepare.prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ),
        patch(
            "neuralls.composition.assignments.comparison_prepare.resolve_comparison_config",
            return_value=mock_cfg(),
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.has_complete_reusable_session",
            return_value=False,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.execute_all",
            return_value=(),
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.outcomes_from_workloads",
            return_value=[
                ComparisonReused(comparison_id="a", run=comparison_run),
                ComparisonReused(comparison_id="b", run=comparison_run),
            ],
        ),
        patch(MLFLOW_MODULE) as mock_mlflow,
    ):
        outcomes = run_loaded_comparison_batch(
            case,
            VIRTUAL_CASE_PATH,
            ComparisonParams(),
            virtual_settings(),
            tracking=fake_tracking,
        )

    assert [outcome.comparison_id for outcome in outcomes] == ["a", "b"]
    mock_mlflow.start_run.assert_called_once()


@pytest.mark.usefixtures("stub_input_gate", "stub_planning", "stub_setup_tracking")
def test_run_comparison_batch_records_and_reraises_blocking_execution_failure(
    prepared_presentation: PreparedComparisonPresentation,
    fake_tracking: MlflowTrackingServices,
) -> None:
    case = in_memory_case(with_comparisons=True)
    original = RuntimeError("injected method failure")
    failure_sink = MagicMock()

    def prepare(
        cfg: object,
        planned: PlannedSystem,
        context: ComparisonResolutionContext,
        *,
        force: bool = False,
    ) -> PreparedComparisonExecution:
        del force
        return PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=contextlib.ExitStack(),
            resolved=ResolvedComparisonSpecs(
                specs=[StandardPreconditionerConfig(type=PreconditionerType.JACOBI)],
                warnings=(),
                checkpoint_digests={},
            ),
            origins_by_method={"jacobi": NoFitData()},
            training_run_costs={},
            identity=StageIdentity.build(
                "comparison", {"entry": canonical_digest(planned.entry.id)}
            ),
            settings=context.settings,
            presentation=prepared_presentation,
            title=prepared_presentation.scope,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_prepare.prepare_comparison_entry",
            side_effect=prepare,
        ),
        patch(
            "neuralls.composition.assignments.comparison_prepare.resolve_comparison_config",
            return_value=mock_cfg(),
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.has_complete_reusable_session",
            return_value=False,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.execute_all",
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
        run_loaded_comparison_batch(
            case,
            VIRTUAL_CASE_PATH,
            ComparisonParams(),
            virtual_settings(),
            tracking=fake_tracking,
        )

    assert raised.value is original
    context = failure_sink.record.call_args.args[0]
    assert context == ComparisonBatchFailureContext(
        phase=ComparisonBatchPhase.METHOD_WORKLOADS,
        case_config_path=VIRTUAL_CASE_PATH,
    )
