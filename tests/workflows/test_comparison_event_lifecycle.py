"""Method-workload execution records canonical setup and solve lifecycles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    MatrixWorkload,
    MatrixWorkloadMember,
    ResolvedMethod,
)
from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models import (
    MeasuredPeakMemory,
    SetupCompleted,
    SetupFailed,
    SolveFailed,
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.workload import (
    CompleteWorkloadSummary,
    IncompleteWorkloadSummary,
)
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, SystemId
from tests.presentation_support import PreparedMethodFactory
from tests.recipe_support import GAUSSIAN

type CapturedLog = Callable[[str], list[str]]


@pytest.fixture
def solver_params() -> SolverParams:
    return SolverParams(
        rtol=1e-6,
        atol=1e-10,
        max_iterations=20,
        stopping_criterion="residual_norm",
        m_max=5,
    )


@pytest.fixture
def one_system_bundle() -> MatrixWorkload:
    tag = SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(1), 0, SystemId("ds-1-0"))
    return MatrixWorkload(
        dataset=DatasetId("ds"),
        matrix_id=MatrixId(1),
        matrix=np.diag(np.array([2.0, 3.0])),
        matrix_path=Path("/workspace/not-a-dataset"),
        matrix_index=0,
        members=(MatrixWorkloadMember(tag=tag, rhs=np.ones(2)),),
    )


@pytest.fixture
def two_matrix_bundles() -> tuple[MatrixWorkload, MatrixWorkload]:
    bundles: list[MatrixWorkload] = []
    for matrix_id in (1, 2):
        members = tuple(
            MatrixWorkloadMember(
                tag=SystemTag(
                    DatasetId("ds"),
                    GAUSSIAN,
                    MatrixId(matrix_id),
                    ordinal,
                    SystemId(f"ds-{matrix_id}-{ordinal}"),
                ),
                rhs=np.array([1.0, float(ordinal + 1)]),
            )
            for ordinal in (0, 1)
        )
        bundles.append(
            MatrixWorkload(
                dataset=DatasetId("ds"),
                matrix_id=MatrixId(matrix_id),
                matrix=np.diag(np.array([2.0, 3.0])) * matrix_id,
                matrix_path=Path("/workspace") / f"matrix-{matrix_id}",
                matrix_index=matrix_id,
                members=members,
            )
        )
    return bundles[0], bundles[1]


def _solved_result(name: str, *, seconds: float = 2.0) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.ones(2),
        converged=True,
        iterations=3,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 1e-8],
        residual_history_abs=[1.0, 1e-8],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        solve_time_seconds=seconds,
        solve_peak_memory_bytes=64,
        solve_time_samples=(1.0, 2.0, 3.0),
    )


def test_workload_models_are_frozen_and_slotted(
    one_system_bundle: MatrixWorkload, prepared_method: PreparedMethodFactory
) -> None:
    method = ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.IDENTITY))
    tag = one_system_bundle.members[0].tag
    completed = CompletedMethodWorkload(
        method=method.name,
        planned=(tag,),
        setup_events=(),
        solve_attempts=(),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(0),
        config_digest="sha256:" + "0" * 64,
        method_config={"name": str(method.name)},
        presentation=ExecutedMethod(prepared_method(str(method.name)), ()),
    )

    for value in (one_system_bundle.members[0], one_system_bundle, method, completed):
        assert not hasattr(value, "__dict__")
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            value.extra = "invalid"  # type: ignore[attr-defined]


def test_setup_failure_is_an_incomplete_zero_attempt_workload(
    monkeypatch: pytest.MonkeyPatch,
    one_system_bundle: MatrixWorkload,
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
    captured_log: CapturedLog,
) -> None:
    """The executor must retain a failed setup without inventing solve attempts."""
    from neuralls.composition.comparison import comparison_run
    from neuralls.composition.comparison.comparison_run import execute_method_workload

    def fail_setup(*args: object, **kwargs: object) -> object:
        raise RuntimeError("setup broke")

    monkeypatch.setattr(comparison_run, "_build_scheduled", fail_setup)
    method = ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.IDENTITY))

    messages = captured_log("WARNING")
    workload = execute_method_workload(
        method,
        (one_system_bundle,),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=solver_params,
        settings=neuralls_settings,
        prepared=prepared_method(str(method.name)),
    )

    assert len(workload.setup_events) == 1
    failed_setup = workload.setup_events[0]
    assert isinstance(failed_setup.outcome, SetupFailed)
    assert isinstance(failed_setup.peak_memory, MeasuredPeakMemory)
    assert failed_setup.samples_s
    assert workload.solve_attempts == ()
    assert workload.planned == (one_system_bundle.members[0].tag,)
    assert len(workload.diagnostics) == 1
    failed_diagnostic = workload.diagnostics[0][1]
    assert failed_diagnostic.preconditioner == method.name
    assert failed_diagnostic.converged is False
    assert failed_diagnostic.error is not None
    summary = summarize_method_workload(
        method.name,
        workload.planned,
        workload.setup_events,
        workload.solve_attempts,
        workload.workload_peak_memory,
    )
    assert isinstance(summary, IncompleteWorkloadSummary)
    assert summary.n_attempted_rhs == 0
    assert len(messages) == 1
    assert messages == [
        (
            "Recoverable setup failure for Jacobi on Test problem | Gaussian CG Residuals "
            "(k=20, N=100), matrix 1: RuntimeError: setup broke"
        )
    ]


@pytest.mark.parametrize(
    "preconditioner_type",
    (PreconditionerType.IDENTITY, PreconditionerType.JACOBI),
)
def test_one_setup_per_matrix_reuses_exact_state_only_for_sibling_rhs(
    monkeypatch: pytest.MonkeyPatch,
    two_matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    preconditioner_type: PreconditionerType,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    """Static and matrix-dependent methods both get fresh state per matrix."""
    from neuralls.composition.comparison import comparison_run

    states: list[object] = []

    def solve(
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        *,
        preconditioners: dict[str, object],
        **kwargs: object,
    ) -> dict[str, CGComparisonResult]:
        del matrix, rhs, kwargs
        name, state = next(iter(preconditioners.items()))
        states.append(state)
        return {name: _solved_result(name)}

    monkeypatch.setattr(comparison_run, "run_cg_comparison", solve)
    monkeypatch.setattr(comparison_run, "compute_reference_solution", lambda *a, **k: None)
    method = ResolvedMethod(StandardPreconditionerConfig(type=preconditioner_type))

    workload = comparison_run.execute_method_workload(
        method,
        two_matrix_bundles,
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=solver_params,
        settings=neuralls_settings,
        prepared=prepared_method(str(method.name)),
    )

    assert len(workload.setup_events) == 2
    assert all(isinstance(event.outcome, SetupCompleted) for event in workload.setup_events)
    assert len(workload.solve_attempts) == 4
    assert states[0] is states[1]
    assert states[2] is states[3]
    assert states[0] is not states[2]


def test_repeat_samples_are_retained_but_totals_use_one_representative(
    monkeypatch: pytest.MonkeyPatch,
    two_matrix_bundles: tuple[MatrixWorkload, MatrixWorkload],
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    from neuralls.composition.comparison import comparison_run

    monkeypatch.setattr(
        comparison_run,
        "run_cg_comparison",
        lambda *a, preconditioners, **k: {
            next(iter(preconditioners)): _solved_result(next(iter(preconditioners)))
        },
    )
    monkeypatch.setattr(comparison_run, "compute_reference_solution", lambda *a, **k: None)
    method = ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.IDENTITY))
    workload = comparison_run.execute_method_workload(
        method,
        two_matrix_bundles,
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=solver_params,
        settings=neuralls_settings,
        prepared=prepared_method(str(method.name)),
    )

    assert all(attempt.samples_s == (1.0, 2.0, 3.0) for attempt in workload.solve_attempts)
    summary = summarize_method_workload(
        method.name,
        workload.planned,
        workload.setup_events,
        workload.solve_attempts,
        workload.workload_peak_memory,
    )
    assert isinstance(summary, CompleteWorkloadSummary)
    assert summary.solve_total_s == 8.0
    assert summary.solve_total_s != sum(
        sample for attempt in workload.solve_attempts for sample in attempt.samples_s
    )


def test_workload_memory_observer_does_not_reset_operation_peak_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuralls.shared import device

    samples = iter((100, 130))
    monkeypatch.setattr(device, "_sample_peak_rss_kib", lambda: next(samples))

    observer = device.WorkloadMemoryObserver.start(torch.device("cpu"))
    observer.observe_operation(5_000)

    assert observer.finish() == 30 * 1024


def test_io_reference_and_reporting_stay_outside_online_timers(
    monkeypatch: pytest.MonkeyPatch,
    one_system_bundle: MatrixWorkload,
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    """Only preparation/binding and the solver occupy online timer regions."""
    from neuralls.composition.comparison import comparison_run
    from neuralls.shared.device import TimedRuns

    trace: list[str] = []

    monkeypatch.setattr(
        comparison_run,
        "_load_extra_inputs",
        lambda *a, **k: trace.append("data-read") or (),
    )
    monkeypatch.setattr(
        comparison_run.PreconditionerService,
        "preload",
        lambda *a, **k: trace.append("checkpoint-read"),
    )

    def timed_setup(
        run: Callable[[], object], *args: object, **kwargs: object
    ) -> TimedRuns[object]:
        del args, kwargs
        trace.append("setup-timer:start")
        value = run()
        trace.append("setup-timer:end")
        return TimedRuns(value, 1.0, (0.5, 1.5), 10)

    monkeypatch.setattr(comparison_run, "repeat_timed_outcome", timed_setup)
    build = comparison_run._build_scheduled_and_bind

    def build_and_bind(*args: object, **kwargs: object) -> object:
        trace.append("convert-bind")
        return build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(comparison_run, "_build_scheduled_and_bind", build_and_bind)
    monkeypatch.setattr(
        comparison_run,
        "compute_reference_solution",
        lambda *a, **k: trace.append("reference") or None,
    )

    def solve(*args: object, preconditioners: dict[str, object], **kwargs: object) -> object:
        del args, kwargs
        trace.extend(("solve-timer:start", "solver", "solve-timer:end"))
        name = next(iter(preconditioners))
        return {name: _solved_result(name)}

    monkeypatch.setattr(comparison_run, "run_cg_comparison", solve)
    method = ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.IDENTITY))

    comparison_run.execute_method_workload(
        method,
        (one_system_bundle,),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=solver_params,
        settings=neuralls_settings,
        prepared=prepared_method(str(method.name)),
    )

    assert trace == [
        "data-read",
        "checkpoint-read",
        "setup-timer:start",
        "convert-bind",
        "setup-timer:end",
        "reference",
        "solve-timer:start",
        "solver",
        "solve-timer:end",
    ]


def test_solver_failure_retains_measured_attempt(
    monkeypatch: pytest.MonkeyPatch,
    one_system_bundle: MatrixWorkload,
    solver_params: SolverParams,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
    captured_log: CapturedLog,
) -> None:
    from neuralls.composition.comparison import comparison_run
    from neuralls.domain.solver import comparison as solver_comparison

    monkeypatch.setattr(comparison_run, "compute_reference_solution", lambda *a, **k: None)

    def fail_solve(*args: object, **kwargs: object) -> object:
        raise RuntimeError("solver broke")

    monkeypatch.setattr(solver_comparison, "_solve_one", fail_solve)
    method = ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.IDENTITY))

    messages = captured_log("WARNING")
    workload = comparison_run.execute_method_workload(
        method,
        (one_system_bundle,),
        datasets=group_datasets,
        notation=NotationPolicy.default(),
        params=solver_params,
        settings=neuralls_settings,
        prepared=prepared_method(str(method.name)),
    )

    assert len(workload.solve_attempts) == 1
    attempt = workload.solve_attempts[0]
    assert isinstance(attempt.outcome, SolveFailed)
    assert isinstance(attempt.peak_memory, MeasuredPeakMemory)
    assert attempt.samples_s
    assert len(messages) == 1
