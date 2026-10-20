"""Batch-level lifetimes: one device warm-up, bounded caches, and one shared lease manager."""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from inspect import isgeneratorfunction
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from neuralls.composition.assignments import comparison_batch
from neuralls.composition.assignments.comparison_batch import (
    _ComparisonResolutionContext,
    _execute_all,
    _method_major_schedule,
    _prepare_comparison_entry,
    _PreparedComparisonExecution,
)
from neuralls.composition.comparison.models import CompletedMethodWorkload, ResolvedMethod
from neuralls.domain.solver.models import MeasuredPeakMemory
from neuralls.domain.solver.models.config import SolverParams
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.types import DatasetId, MatrixId


def _system(dataset: str, matrix: int, entry_id: str = "system") -> MagicMock:
    system = MagicMock(spec=_PreparedComparisonExecution)
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
    return system


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

    with pytest.raises(ValueError, match="different configuration identities"):
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

    with pytest.raises(ValueError, match="incompatible matrix/solver semantics"):
        _method_major_schedule((first, second))


def test_method_schedule_rejects_different_method_membership_on_siblings() -> None:
    first = _system("ds", 0, "r0")
    second = _system("ds", 0, "r1")
    identity = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    first.resolved.specs = [identity, jacobi]
    second.resolved.specs = [identity]

    with pytest.raises(ValueError, match="different method membership"):
        _method_major_schedule((first, second))


def test_matrix_workloads_are_loaded_lazily_per_method_matrix() -> None:
    assert isgeneratorfunction(comparison_batch._load_matrix_workloads)


def test_matrix_loader_releases_prepared_system_before_next_matrix_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The producer helper must not retain its prior prepared system at ``next()``."""
    first = _system("ds", 0, "r0")
    second = _system("ds", 1, "r1")
    for system, matrix_id in ((first, 0), (second, 1)):
        system.cfg.general.data.matrix_path = tmp_path / f"matrix-{matrix_id}"
        system.cfg.general.data.matrix_index = 0
        system.cfg.general.data.require_non_residual_rhs = True
        system.cfg.general.data.selection_seed = None
        system.cfg.general.data.rhs_source_kind = "gaussian"
        system.cfg.general.data.rhs_source_params = {}
        system.entry.rhs_source.recipe_label = "gaussian"
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


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(comparison_batch, "warm_up_device", lambda device: recorded.append("warm"))
    monkeypatch.setattr(comparison_batch, "release_device_memory", lambda: None)
    monkeypatch.setattr(comparison_batch.mlflow, "log_metric", lambda *a, **k: None)

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
            offline_costs=(),
            workload_peak_memory=MeasuredPeakMemory(0),
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

    outcomes, workloads = _execute_all(pending)

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
    assert len(outcomes) == 4
    assert len(workloads) == 2


def test_every_entry_of_a_sweep_shares_the_one_lease_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leases_seen: list[object] = []

    def resolve(cfg: object, context: object, artifact_leases: object) -> None:
        leases_seen.append(artifact_leases)
        raise ValueError("stop after resolution")

    monkeypatch.setattr(comparison_batch, "_validate_comparison_sources", lambda cfg: None)
    monkeypatch.setattr(comparison_batch, "_resolve_comparison_specs", resolve)
    monkeypatch.setattr(comparison_batch, "_comparison_failure_outcome", lambda entry, exc: exc)
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
