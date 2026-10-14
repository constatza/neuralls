"""Batch-level lifetimes: one device warm-up, bounded caches, and one shared lease manager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuralls.composition.assignments import comparison_batch
from neuralls.composition.assignments.comparison_batch import (
    _ComparisonResolutionContext,
    _execute_all,
    _prepare_comparison_entry,
    _PreparedComparisonExecution,
)
from neuralls.composition.comparison._preconditioner_setup import (
    DatasetPreconditionerCache,
    DatasetScope,
)
from neuralls.shared.types import MatrixId


def _system(dataset: str, matrix: int) -> MagicMock:
    system = MagicMock(spec=_PreparedComparisonExecution)
    system.entry = MagicMock()
    system.planned = MagicMock()
    system.entry.matrix_dataset = dataset
    system.planned.matrix.matrix_id = MatrixId(matrix)
    return system


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(comparison_batch, "warm_up_device", lambda device: recorded.append("warm"))
    monkeypatch.setattr(comparison_batch, "release_device_memory", lambda: None)
    monkeypatch.setattr(comparison_batch.mlflow, "log_metric", lambda *a, **k: None)

    def execute(prepared: MagicMock, *, cache: object) -> str:
        recorded.append(
            f"run:{prepared.entry.matrix_dataset}:{int(prepared.planned.matrix.matrix_id)}"
        )
        return "outcome"

    monkeypatch.setattr(comparison_batch, "_execute_prepared_comparison", execute)
    evict = DatasetPreconditionerCache.evict_matrix
    release = DatasetScope.release

    def spy_evict(self: DatasetPreconditionerCache, matrix: MatrixId) -> None:
        recorded.append(f"evict:{int(matrix)}")
        evict(self, matrix)

    def spy_release(self: DatasetScope) -> None:
        recorded.append("release")
        release(self)

    monkeypatch.setattr(DatasetPreconditionerCache, "evict_matrix", spy_evict)
    monkeypatch.setattr(DatasetScope, "release", spy_release)
    return recorded


def test_device_is_warmed_once_and_state_is_dropped_after_each_last_use(events: list[str]) -> None:
    pending = [_system("a", 0), _system("a", 0), _system("a", 1), _system("b", 0)]

    outcomes, caches = _execute_all(pending)

    assert events == [
        "warm",
        "run:a:0",
        "run:a:0",
        "evict:0",
        "run:a:1",
        "evict:1",
        "release",
        "run:b:0",
        "evict:0",
        "release",
    ]
    assert outcomes == ["outcome"] * 4
    assert set(caches) == {"a", "b"}


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
