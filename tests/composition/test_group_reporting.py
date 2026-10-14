"""``report_all_levels``: which report groups reach the sink, how they are described,
and what is logged otherwise."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from loguru import logger

from neuralls.composition.comparison._preconditioner_setup import (
    DatasetPreconditionerCache,
    PreconditionerService,
)
from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    MatrixRef,
    PlannedSystem,
)
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonResult,
    GroupAggregated,
    GroupPresentation,
    PreconditionerPlotStyle,
)
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.types import DatasetId, MatrixId

JACOBI = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
JACOBI_STYLE = PreconditionerPlotStyle(label="Jacobi (D⁻¹)", family=PreconditionerType.JACOBI)
OMEGA = 6.283185307179586


@dataclass
class RecordingSink:
    """Stands in for the MLflow sink: records what it was asked to report."""

    outcomes: list[GroupAggregated] = field(default_factory=list)
    presentations: list[GroupPresentation] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        self.outcomes.append(outcome)
        self.presentations.append(presentation)

    @property
    def group_ids(self) -> list[str]:
        return [outcome.group_id for outcome in self.outcomes]


def _system(
    entry_id: str,
    *,
    dataset: str = "ds",
    matrix: int = 0,
    ordinal: int = 0,
    rhs_source: dict[str, object] | None = None,
) -> PlannedSystem:
    return PlannedSystem(
        entry=ComparisonRegistryEntry(
            id=entry_id,
            matrix_dataset=dataset,
            rhs_source=rhs_source or {"kind": "gaussian"},
        ),
        matrix=MatrixRef(matrix_id=MatrixId(matrix), row=matrix),
        rhs_row=None,
        rhs_ordinal=ordinal,
        provenance=None,
    )


def _cg_result(name: str) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=4,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 1e-9],
        residual_history_abs=[1.0, 1e-9],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=1.0,
        solve_time_seconds=0.5,
    )


def _executed(entry_id: str, spec: StandardPreconditionerConfig = JACOBI) -> ComparisonOutcome:
    general = ComparisonGeneral(
        params=SolverParams(
            rtol=1e-6, atol=1e-14, max_iterations=10, stopping_criterion="residual_norm", m_max=1
        ),
        data=ComparisonData(matrix_path=None, rhs_path=None),
    )
    payload = ComparisonResult(
        results={spec.name: _cg_result(spec.name)},
        summary="",
        solver_params=general,
        plot_styles={spec.name: JACOBI_STYLE},
    )
    return ComparisonOutcome(entry_id, entry_id, success=True, payload=payload)


def _cache_hit(entry_id: str) -> ComparisonOutcome:
    return ComparisonOutcome(entry_id, entry_id, success=True, payload=None)


def _failed(entry_id: str) -> ComparisonOutcome:
    return ComparisonOutcome(entry_id, entry_id, success=False, error="boom")


def _report(
    planned: list[PlannedSystem],
    outcomes: list[ComparisonOutcome],
    sink: RecordingSink,
) -> None:
    report_all_levels(
        outcomes,
        planned,
        {outcome.comparison_id: [JACOBI] for outcome in outcomes if outcome.payload is not None},
        {},
        sink,
    )


@pytest.fixture
def two_dataset_case() -> list[PlannedSystem]:
    """`ds`: matrices 3 and 7, two gaussian draws each plus one sinusoidal; `other`: one system."""
    sinus = {"kind": "sinusoidal", "omega": OMEGA}
    return [
        _system("a-3-0", matrix=3, ordinal=0),
        _system("a-3-1", matrix=3, ordinal=1),
        _system("a-7-0", matrix=7, ordinal=0),
        _system("a-7-1", matrix=7, ordinal=1),
        _system("s-3", matrix=3, rhs_source=sinus),
        _system("o-0", dataset="other"),
    ]


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    yield messages
    logger.remove(handler_id)


def test_every_level_reports_its_groups_finest_level_first(
    two_dataset_case: list[PlannedSystem],
) -> None:
    sink = RecordingSink()

    _report(two_dataset_case, [_executed(s.entry.id) for s in two_dataset_case], sink)

    assert sink.group_ids == [
        "fine/ds/gaussian",
        "fine/ds/sinusoidal_omega_6.28319",
        "fine/other/gaussian",
        "medium/ds",
        "medium/other",
        "global",
    ]
    assert len(set(sink.group_ids)) == len(sink.group_ids)


def test_medium_and_global_groups_pool_systems_and_say_mixture(
    two_dataset_case: list[PlannedSystem],
) -> None:
    sink = RecordingSink()

    _report(two_dataset_case, [_executed(s.entry.id) for s in two_dataset_case], sink)

    by_id = dict(zip(sink.group_ids, sink.presentations, strict=True))
    assert "mixture of gaussian, sinusoidal ω=6.28319" in by_id["medium/ds"].subject
    assert by_id["global"].subject.startswith("all datasets, mixture of ")
    assert "mixture" not in by_id["fine/ds/gaussian"].subject
    assert by_id["fine/ds/gaussian"].subject == "ds, gaussian"


def test_subtitle_states_level_keys_recipe_and_system_count(
    two_dataset_case: list[PlannedSystem],
) -> None:
    sink = RecordingSink()

    _report(two_dataset_case, [_executed(s.entry.id) for s in two_dataset_case], sink)

    by_id = dict(zip(sink.group_ids, sink.presentations, strict=True))
    assert by_id["fine/ds/gaussian"].subtitle == (
        "Fine level · keys: dataset, recipe · recipe: gaussian · N = 4 (2 matrices × RHS)"
    )
    assert by_id["medium/ds"].subtitle.startswith("Medium level · keys: dataset · mixture of ")
    assert by_id["medium/ds"].subtitle.endswith("N = 5 (2 matrices × RHS)")
    assert by_id["global"].subtitle.startswith("Global level · keys: none · mixture of ")
    assert by_id["global"].subtitle.endswith("N = 6 (3 matrices × RHS)")


def test_scatter_labels_name_matrix_and_rhs_ordinal(
    two_dataset_case: list[PlannedSystem],
) -> None:
    sink = RecordingSink()

    _report(two_dataset_case, [_executed(s.entry.id) for s in two_dataset_case], sink)

    by_id = dict(zip(sink.group_ids, sink.presentations, strict=True))
    assert by_id["fine/ds/gaussian"].system_labels == {
        "a-3-0": "m3/r0",
        "a-3-1": "m3/r1",
        "a-7-0": "m7/r0",
        "a-7-1": "m7/r1",
    }
    assert by_id["medium/ds"].system_labels["s-3"].startswith("m3/r0 sinusoidal")
    assert by_id["global"].system_labels["o-0"] == "other m0/r0 gaussian"


def test_plot_styles_come_from_the_run_results(two_dataset_case: list[PlannedSystem]) -> None:
    sink = RecordingSink()

    _report(two_dataset_case, [_executed(s.entry.id) for s in two_dataset_case], sink)

    assert all(p.plot_styles == {JACOBI.name: JACOBI_STYLE} for p in sink.presentations)


def test_a_single_system_group_is_reported_without_a_spread() -> None:
    sink = RecordingSink()

    _report([_system("only")], [_executed("only")], sink)

    assert len(sink.outcomes) == 3
    projected = sink.outcomes[0].projected[JACOBI.name]
    assert projected.n_systems == 1
    assert projected.solve_time_std_seconds is None


def test_all_cache_hit_batch_logs_loudly_and_touches_no_destination(
    two_dataset_case: list[PlannedSystem], warnings_log: list[str]
) -> None:
    sink = RecordingSink()

    _report(two_dataset_case, [_cache_hit(s.entry.id) for s in two_dataset_case], sink)

    assert sink.outcomes == []
    assert any("global" in message and "nothing measured" in message for message in warnings_log)
    assert any("fine/ds/gaussian" in message for message in warnings_log)


def test_a_partially_measured_group_is_reported_with_a_warning(
    warnings_log: list[str],
) -> None:
    planned = [_system(f"s{i}", ordinal=i) for i in range(4)]
    outcomes = [_executed("s0"), _executed("s1"), _cache_hit("s2"), _failed("s3")]
    sink = RecordingSink()

    _report(planned, outcomes, sink)

    aggregated = sink.outcomes[0]
    assert (aggregated.n_aggregated, aggregated.n_in_group) == (2, 4)
    assert any("aggregated 2 of 4" in m and "1 cache hit(s), 1 failed" in m for m in warnings_log)


def test_a_fully_cached_recipe_is_skipped_while_its_sibling_recipe_is_reported(
    warnings_log: list[str],
) -> None:
    planned = [_system("g"), _system("s", rhs_source={"kind": "sinusoidal", "omega": OMEGA})]
    sink = RecordingSink()

    _report(planned, [_executed("g"), _cache_hit("s")], sink)

    assert sink.group_ids == ["fine/ds/gaussian", "medium/ds", "global"]
    assert any("fine/ds/sinusoidal_omega_6.28319" in m for m in warnings_log)


def test_preconditioners_present_in_only_some_systems_still_get_a_sharing_policy() -> None:
    planned = [_system("a"), _system("b", ordinal=1)]
    extra = StandardPreconditionerConfig(type=PreconditionerType.ILU)
    first, second = _executed("a"), _executed("b")
    assert second.payload is not None
    second.payload.results[extra.name] = _cg_result(extra.name)
    sink = RecordingSink()

    report_all_levels([first, second], planned, {"a": [JACOBI], "b": [JACOBI, extra]}, {}, sink)

    assert set(sink.outcomes[0].projected) == {JACOBI.name, extra.name}


def test_a_static_setup_name_shared_in_its_dataset_projects_as_shared() -> None:
    identity = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    planned = [_system("a"), _system("b", ordinal=1)]
    cache = DatasetPreconditionerCache()
    cache.for_system(MatrixId(0)).put(
        identity.name,
        PreconditionerService().create_preconditioner(torch.eye(2, dtype=torch.float64), identity),
        setup_cost_seconds=1.0,
        setup_peak_memory_bytes=None,
    )
    sink = RecordingSink()

    report_all_levels(
        [_executed("a", identity), _executed("b", identity)],
        planned,
        {"a": [identity], "b": [identity]},
        {DatasetId("ds"): cache},
        sink,
    )

    assert sink.outcomes[0].projected[identity.name].setup_sharing.share_setup is True


def test_a_name_unshared_in_any_dataset_projects_as_unshared() -> None:
    planned = [_system("a"), _system("b", dataset="other")]
    sink = RecordingSink()

    report_all_levels(
        [_executed("a"), _executed("b")],
        planned,
        {"a": [JACOBI], "b": [JACOBI]},
        {DatasetId("ds"): DatasetPreconditionerCache()},
        sink,
    )

    assert sink.outcomes[-1].projected[JACOBI.name].setup_sharing.share_setup is False
