"""Batch-level lifetimes: one device warm-up, bounded caches, and one shared lease manager."""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Sequence
from inspect import isgeneratorfunction
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from neuralls.composition.assignments import comparison_batch
from neuralls.composition.assignments.comparison_batch import (
    PreparedComparisonBatch,
    PreparedComparisonExecution,
    _ComparisonResolutionContext,
    _execute_all,
    _method_major_schedule,
    _prepare_comparison_entry,
)
from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    MatrixWorkload,
    ResolvedMethod,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    MethodPresentationCandidate,
    PreparedMethodPresentation,
    StandaloneMethodPresentation,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models import MeasuredPeakMemory
from neuralls.domain.solver.models.config import SolverParams
from neuralls.platform.config.models.comparison import GaussianRhsSourceModel
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName
from neuralls.shared.workflow_failures import (
    ComparisonMatrixFailureContext,
    ComparisonMatrixPhase,
    ComparisonMethodFailureContext,
)
from tests.presentation_support import PreparedMethodFactory


def _system(dataset: str, matrix: int, entry_id: str = "system") -> MagicMock:
    system = MagicMock(spec=PreparedComparisonExecution)
    system.entry = MagicMock()
    system.planned = MagicMock()
    system.entry.matrix_dataset = dataset
    system.entry.id = entry_id
    system.planned.matrix.matrix_id = MatrixId(matrix)
    system.__dict__["resolved"] = MagicMock()
    system.resolved.checkpoint_digests = {}
    system.__dict__["cfg"] = MagicMock()
    system.cfg.general.params = SolverParams(
        rtol=1e-6,
        atol=1e-10,
        max_iterations=20,
        stopping_criterion="residual_norm",
        m_max=5,
    )
    system.cfg.general.data.normalize_system = "matrix"
    system.__dict__["settings"] = MagicMock()
    system.cfg.notation_policy = NotationPolicy.default()
    system.__dict__["presentation"] = MagicMock()
    system.presentation.method.side_effect = lambda name: PreparedMethodPresentation(
        PreconditionerName(name),
        StandaloneMethodPresentation(ScientificDescription("Jacobi")),
        (),
    )
    system.__dict__["identity"] = StageIdentity.build(
        "comparison", {"entry": canonical_digest(entry_id)}
    )
    return system


def _batch(systems: list[MagicMock]) -> PreparedComparisonBatch:
    """Wrap fake prepared executions in the batch type execution requires."""
    return PreparedComparisonBatch(
        items=tuple(systems), planned=tuple(system.planned for system in systems)
    )


def test_method_major_schedule_nests_matrix_and_rhs_in_plan_order() -> None:
    identity = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    pending = [
        _system("ds", 1, "m1-r0"),
        _system("ds", 1, "m1-r1"),
        _system("ds", 2, "m2-r0"),
        _system("ds", 2, "m2-r1"),
    ]
    for system in pending:
        system.resolved.specs = [identity, jacobi]

    schedule = _method_major_schedule(pending)
    trace = [
        (method.name, matrix_id, tuple(system.entry.id for system in systems))
        for method, matrices in schedule
        for (_, matrix_id), systems in matrices
    ]

    assert trace == [
        ("identity", MatrixId(1), ("m1-r0", "m1-r1")),
        ("identity", MatrixId(2), ("m2-r0", "m2-r1")),
        ("jacobi", MatrixId(1), ("m1-r0", "m1-r1")),
        ("jacobi", MatrixId(2), ("m2-r0", "m2-r1")),
    ]


def test_method_schedule_does_not_share_same_name_with_different_identity() -> None:
    first = _system("a", 0, "a-0")
    second = _system("b", 0, "b-0")
    first.resolved.specs = [StandardPreconditionerConfig(type=PreconditionerType.JACOBI)]
    second.resolved.specs = [
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI, start_iter=5)
    ]

    with pytest.raises(ValueError):
        _method_major_schedule((first, second))


def test_method_schedule_rejects_mixed_matrix_and_solver_semantics() -> None:
    first = _system("ds", 0, "r0")
    second = _system("ds", 0, "r1")
    config = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    for system in (first, second):
        system.resolved.specs = [config]
        system.cfg.general.params = SolverParams(
            rtol=1e-6,
            atol=1e-10,
            max_iterations=20,
            stopping_criterion="residual_norm",
            m_max=5,
        )
        system.cfg.general.data.normalize_system = "matrix"
    second.cfg.general.data.normalize_system = "rhs"

    with pytest.raises(ValueError):
        _method_major_schedule((first, second))


def test_method_schedule_rejects_different_method_membership_on_siblings() -> None:
    first = _system("ds", 0, "r0")
    second = _system("ds", 0, "r1")
    identity = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    first.resolved.specs = [identity, jacobi]
    second.resolved.specs = [identity]

    with pytest.raises(ValueError):
        _method_major_schedule((first, second))


def test_matrix_workloads_are_loaded_lazily_per_method_matrix() -> None:
    assert isgeneratorfunction(comparison_batch._load_matrix_workloads)


def test_matrix_loader_releases_prepared_system_before_next_matrix_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer helper must not retain its prior prepared system at ``next()``."""
    first = _system("ds", 0, "r0")
    second = _system("ds", 1, "r1")
    for system, matrix_id in ((first, 0), (second, 1)):
        system.cfg.general.data.matrix_path = Path("/workspace") / f"matrix-{matrix_id}"
        system.cfg.general.data.matrix_index = 0
        system.cfg.general.data.require_non_residual_rhs = True
        system.cfg.general.data.selection_seed = None
        system.cfg.general.data.rhs_source_kind = "gaussian"
        system.cfg.general.data.rhs_source_params = {}
        system.entry.rhs_source = GaussianRhsSourceModel()
        system.planned.entry = system.entry
        system.planned.matrix.matrix_id = MatrixId(matrix_id)
        system.planned.rhs_ordinal = 0
        system.resolved.specs = [StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)]

    prepared_refs: list[weakref.ReferenceType[object]] = []
    load_count = 0

    class PreparedSystem:
        def __init__(self) -> None:
            self.matrix = torch.eye(2)
            self.rhs = torch.ones(2)

    def resolve(**_: object) -> object:
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            assert prepared_refs[0]() is None
        return object()

    def prepare(*_: object) -> PreparedSystem:
        prepared = PreparedSystem()
        prepared_refs.append(weakref.ref(prepared))
        return prepared

    monkeypatch.setattr(comparison_batch, "resolve_comparison_input", resolve)
    monkeypatch.setattr(comparison_batch, "prepare_resolved_linear_system", prepare)
    matrices = (
        ((DatasetId("ds"), MatrixId(0)), (first,)),
        ((DatasetId("ds"), MatrixId(1)), (second,)),
    )
    bundles = comparison_batch._load_matrix_workloads(matrices)

    first_bundle = next(bundles)
    assert prepared_refs[0]() is None
    second_bundle = next(bundles)
    assert first_bundle.matrix.shape == second_bundle.matrix.shape


@pytest.fixture(autouse=True)
def _no_parameter_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests mock the run lifecycle; parameter logging has no run to write to."""
    monkeypatch.setattr(comparison_batch.mlflow, "log_params", lambda *a, **k: None)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch, prepared_method: PreparedMethodFactory) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(comparison_batch, "warm_up_device", lambda device: recorded.append("warm"))
    monkeypatch.setattr(comparison_batch, "release_device_memory", lambda: None)
    monkeypatch.setattr(comparison_batch.mlflow, "log_metric", lambda *a, **k: None)
    child_run = MagicMock()
    child_run.__enter__.return_value = MagicMock()
    child_run.__exit__.return_value = False
    monkeypatch.setattr(comparison_batch.mlflow, "start_run", MagicMock(return_value=child_run))

    monkeypatch.setattr(
        comparison_batch,
        "_load_matrix_workloads",
        lambda matrices: tuple(matrix for matrix, _ in matrices),
    )

    def execute(
        method: ResolvedMethod,
        bundles: Sequence[tuple[object, MatrixId]],
        **kwargs: object,
    ) -> CompletedMethodWorkload:
        del kwargs
        name = method.name
        recorded.append(f"method:{name}")
        recorded.extend(f"matrix:{name}:{int(matrix_id)}" for _, matrix_id in bundles)
        recorded.append(f"release:{name}")
        return CompletedMethodWorkload(
            method=name,
            planned=(),
            setup_events=(),
            solve_attempts=(),
            diagnostics=(),
            generation_cost=None,
            workload_peak_memory=MeasuredPeakMemory(0),
            config_digest="sha256:" + "0" * 64,
            method_config={"name": str(name)},
            presentation=ExecutedMethod(prepared_method(str(name)), ()),
        )

    monkeypatch.setattr(comparison_batch, "execute_method_workload", execute)
    return recorded


def test_device_is_warmed_once_and_each_method_releases_before_the_next(
    events: list[str],
) -> None:
    identity = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    pending = [
        _system("a", 0, "a-0-0"),
        _system("a", 0, "a-0-1"),
        _system("a", 1, "a-1-0"),
        _system("b", 0, "b-0-0"),
    ]
    for system in pending:
        system.resolved.specs = [identity, jacobi]

    comparison_batch.warm_up_device(torch.device("cpu"))
    workloads = _execute_all(
        _batch(pending),
        method_sink=MagicMock(),
        failure_sink=MagicMock(),
    )

    assert events == [
        "warm",
        "method:identity",
        "matrix:identity:0",
        "matrix:identity:1",
        "matrix:identity:0",
        "release:identity",
        "method:jacobi",
        "matrix:jacobi:0",
        "matrix:jacobi:1",
        "matrix:jacobi:0",
        "release:jacobi",
    ]
    assert len(workloads) == 2


def test_method_child_wraps_execution_and_persistence(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = [_system("a", 0, "a-0-0")]
    pending[0].resolved.specs = [StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)]
    run = MagicMock()
    run.__enter__.side_effect = lambda: events.append("child:enter") or MagicMock()
    run.__exit__.side_effect = lambda *args: events.append("child:exit") or False
    monkeypatch.setattr(comparison_batch.mlflow, "start_run", MagicMock(return_value=run))

    class MethodSink:
        def record_method(self, workload: object, summary: object) -> None:
            del workload, summary
            events.append("persist")

    class FailureSink:
        def record(self, context: object, exc: Exception) -> None:
            raise AssertionError(f"unexpected failure: {context}: {exc}")

    _execute_all(_batch(pending), method_sink=MethodSink(), failure_sink=FailureSink())

    assert events.index("child:enter") < events.index("method:identity")
    assert events.index("method:identity") < events.index("persist")
    assert events.index("persist") < events.index("child:exit")


def test_method_failure_reraises_original_when_failure_sink_also_fails(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = [_system("a", 0, "a-0-0")]
    pending[0].resolved.specs = [StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)]
    original = RuntimeError("method exploded")
    monkeypatch.setattr(
        comparison_batch, "execute_method_workload", MagicMock(side_effect=original)
    )
    run = MagicMock()
    run.__enter__.return_value = MagicMock()
    run.__exit__.return_value = False
    monkeypatch.setattr(comparison_batch.mlflow, "start_run", MagicMock(return_value=run))

    class MethodSink:
        def record_method(self, workload: object, summary: object) -> None:
            raise AssertionError("persistence must not run")

    contexts: list[ComparisonMethodFailureContext] = []

    class FailingFailureSink:
        def record(self, context: ComparisonMethodFailureContext, exc: Exception) -> None:
            contexts.append(context)
            raise RuntimeError("diagnostics unavailable")

    with pytest.raises(RuntimeError) as raised:
        _execute_all(
            _batch(pending),
            method_sink=MethodSink(),
            failure_sink=FailingFailureSink(),
        )

    assert raised.value is original
    assert contexts[0].method == "identity"
    assert contexts[0].phase.value == "execution"


def test_matrix_loading_failure_records_exact_matrix_context_once_in_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = [_system("dataset-a", 7, "system-7")]
    pending[0].resolved.specs = [StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)]
    original = RuntimeError("matrix input broke")
    monkeypatch.setattr(
        comparison_batch,
        "_load_one_matrix_workload",
        MagicMock(side_effect=original),
    )
    monkeypatch.setattr(comparison_batch, "release_device_memory", lambda: None)
    child_run = MagicMock()
    child_run.__enter__.return_value = MagicMock()
    child_run.__exit__.return_value = False
    monkeypatch.setattr(comparison_batch.mlflow, "start_run", MagicMock(return_value=child_run))

    def consume_bundles(
        method: ResolvedMethod, bundles: Iterable[MatrixWorkload], **kwargs: object
    ) -> CompletedMethodWorkload:
        del method, kwargs
        tuple(bundles)
        raise AssertionError("the matrix load should fail first")

    monkeypatch.setattr(comparison_batch, "execute_method_workload", consume_bundles)
    contexts: list[object] = []

    class MethodSink:
        def record_method(self, workload: object, summary: object) -> None:
            raise AssertionError("persistence must not run")

    class FailureSink:
        def record(self, context: object, exc: Exception) -> None:
            assert exc is original
            contexts.append(context)

    with pytest.raises(RuntimeError) as raised:
        _execute_all(
            _batch(pending),
            method_sink=MethodSink(),
            failure_sink=FailureSink(),
        )

    assert raised.value is original
    assert contexts == [
        ComparisonMatrixFailureContext(
            phase=ComparisonMatrixPhase.INPUT_LOADING,
            method=PreconditionerName("identity"),
            dataset=DatasetId("dataset-a"),
            matrix_id=MatrixId(7),
        )
    ]


def test_every_entry_of_a_sweep_shares_the_one_lease_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leases_seen: list[object] = []

    def resolve(cfg: object, context: object, artifact_leases: object) -> None:
        leases_seen.append(artifact_leases)
        raise ValueError("stop after resolution")

    monkeypatch.setattr(comparison_batch, "_validate_comparison_sources", lambda cfg: None)
    monkeypatch.setattr(comparison_batch, "_resolve_comparison_specs", resolve)
    monkeypatch.setattr(comparison_batch, "_undescribed_failure", lambda entry, exc: exc)
    shared_leases = MagicMock()
    context = _ComparisonResolutionContext(
        topology=MagicMock(),
        case_config_path=MagicMock(),
        settings=MagicMock(),
        artifact_leases=shared_leases,
    )

    for _ in range(3):
        _prepare_comparison_entry(MagicMock(), MagicMock(), context)

    assert leases_seen == [shared_leases] * 3
    shared_leases.__exit__.assert_not_called()


def test_method_run_forwards_the_cohort_presentation_of_its_own_method(
    events: list[str],
    monkeypatch: pytest.MonkeyPatch,
    dataset_presentation: DatasetPresentation,
) -> None:
    cohort = prepare_presentations(
        UnvalidatedComparisonPresentation(
            methods=tuple(
                MethodPresentationCandidate(
                    PreconditionerName(name),
                    StandaloneMethodPresentation(ScientificDescription(base)),
                )
                for name, base in (("identity", "Identity"), ("jacobi", "Jacobi"))
            ),
            scope=ComparisonScopePresentation(dataset_presentation),
        )
    )
    system = _system("a", 0, "a-0-0")
    system.resolved.specs = [
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
    ]
    system.__dict__["presentation"] = cohort
    forwarded: dict[str, PreparedMethodPresentation] = {}

    def execute(
        method: ResolvedMethod,
        bundles: object,
        *,
        prepared: PreparedMethodPresentation,
        **_: object,
    ) -> CompletedMethodWorkload:
        del bundles
        forwarded[str(method.name)] = prepared
        return CompletedMethodWorkload(
            method=method.name,
            planned=(),
            setup_events=(),
            solve_attempts=(),
            diagnostics=(),
            generation_cost=None,
            workload_peak_memory=MeasuredPeakMemory(0),
            config_digest="sha256:" + "0" * 64,
            method_config={"name": str(method.name)},
            presentation=ExecutedMethod(prepared, ()),
        )

    monkeypatch.setattr(comparison_batch, "execute_method_workload", execute)

    _execute_all(_batch([system]), method_sink=MagicMock(), failure_sink=MagicMock())

    assert forwarded == {
        "identity": cohort.method(PreconditionerName("identity")),
        "jacobi": cohort.method(PreconditionerName("jacobi")),
    }


def test_method_child_is_named_by_its_presentation_and_logs_configured_facts_before_executing(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = [_system("a", 0, "a-0-0")]
    pending[0].resolved.specs = [StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)]
    started: list[dict[str, object]] = []
    run = MagicMock()
    run.__enter__.return_value = MagicMock()
    run.__exit__.return_value = False

    def start_run(**kwargs: object) -> MagicMock:
        started.append(kwargs)
        events.append("child:enter")
        return run

    monkeypatch.setattr(comparison_batch.mlflow, "start_run", start_run)
    monkeypatch.setattr(
        comparison_batch,
        "log_configured_method_facts",
        lambda prepared: events.append(f"facts:{prepared.method_id}"),
    )

    class MethodSink:
        def record_method(self, workload: object, summary: object) -> None:
            del workload, summary

    _execute_all(_batch(pending), method_sink=MethodSink(), failure_sink=MagicMock())

    (call,) = started
    assert call["run_name"] == "Jacobi"
    tags = call["tags"]
    assert isinstance(tags, dict)
    assert tags["method_id"] == "identity"
    assert tags["method_label"] == "Jacobi"
    assert events.index("child:enter") < events.index("facts:identity")
    assert events.index("facts:identity") < events.index("method:identity")
