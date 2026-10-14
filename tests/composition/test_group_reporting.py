"""``report_all_groupings``: which groups reach the sink, and what is logged otherwise."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest

from neuralls.composition.comparison.group_reporting import report_all_groupings
from neuralls.composition.comparison.models import (
    ComparisonGroup,
    ComparisonGroupings,
    ComparisonOutcome,
    GroupKey,
    SingleMatrixDataset,
)
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonResult,
    GroupAggregated,
    ProjectedComparisonResult,
)
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.constants import MIN_GROUP_SIZE
from neuralls.shared.types import ComparisonRhsSourceKind, DatasetId, GroupId

DATASET = DatasetId("ds")
GAUSSIAN = ComparisonRhsSourceKind.GAUSSIAN
SINUSOIDAL = ComparisonRhsSourceKind.SINUSOIDAL
JACOBI = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)


@dataclass
class RecordingSink:
    """Stands in for the MLflow sink: records what it was asked to report."""

    reported: list[tuple[GroupId, Mapping[str, ProjectedComparisonResult], str]] = field(
        default_factory=list
    )
    outcomes: list[GroupAggregated] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, title: str) -> None:
        self.reported.append((outcome.group_id, outcome.projected, title))
        self.outcomes.append(outcome)


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


def _executed(entry_id: str) -> ComparisonOutcome:
    general = ComparisonGeneral(
        params=SolverParams(
            rtol=1e-6, atol=1e-14, max_iterations=10, stopping_criterion="residual_norm", m_max=1
        ),
        data=ComparisonData(matrix_path=None, rhs_path=None),
    )
    payload = ComparisonResult(
        results={JACOBI.name: _cg_result(JACOBI.name)}, summary="", solver_params=general
    )
    return ComparisonOutcome(entry_id, entry_id, success=True, payload=payload)


def _cache_hit(entry_id: str) -> ComparisonOutcome:
    return ComparisonOutcome(entry_id, entry_id, success=True, payload=None)


def _group(rhs_kind: ComparisonRhsSourceKind | None, entry_ids: tuple[str, ...]) -> ComparisonGroup:
    return ComparisonGroup(
        key=GroupKey(dataset_id=DATASET, rhs_kind=rhs_kind),
        matrix_identity=SingleMatrixDataset(dataset_id=DATASET),
        entry_ids=entry_ids,
        rhs_kind_counts=((GAUSSIAN, len(entry_ids)),),
    )


def _groupings(fine: ComparisonGroup, coarse: ComparisonGroup) -> ComparisonGroupings:
    return ComparisonGroupings(
        by_matrix={coarse.group_id: coarse},
        by_matrix_and_rhs_kind={fine.group_id: fine},
    )


@pytest.fixture
def entry_ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MIN_GROUP_SIZE))


def test_a_group_with_enough_results_reaches_the_sink_headline_group_first(
    entry_ids: tuple[str, ...],
) -> None:
    fine, coarse = _group(GAUSSIAN, entry_ids), _group(None, entry_ids)
    sink = RecordingSink()

    report_all_groupings(
        [_executed(e) for e in entry_ids],
        {e: [JACOBI] for e in entry_ids},
        _groupings(fine, coarse),
        {},
        sink,
    )

    assert [group_id for group_id, _, _ in sink.reported] == ["ds::gaussian", "ds"]
    assert set(sink.reported[0][1]) == {JACOBI.name}


def test_a_pooled_group_is_titled_as_a_mixture_with_its_composition(
    entry_ids: tuple[str, ...],
) -> None:
    pooled = ComparisonGroup(
        key=GroupKey(dataset_id=DATASET, rhs_kind=None),
        matrix_identity=SingleMatrixDataset(dataset_id=DATASET),
        entry_ids=entry_ids,
        rhs_kind_counts=((GAUSSIAN, 2), (SINUSOIDAL, 1)),
    )
    sink = RecordingSink()

    report_all_groupings(
        [_executed(e) for e in entry_ids],
        {e: [JACOBI] for e in entry_ids},
        ComparisonGroupings(by_matrix={pooled.group_id: pooled}, by_matrix_and_rhs_kind={}),
        {},
        sink,
    )

    title = sink.reported[0][2]
    assert "pooled mixture" in title
    assert "gaussian ×2" in title
    assert "sinusoidal ×1" in title


def test_a_group_below_the_minimum_size_never_reaches_the_sink() -> None:
    ids = ("only",)
    sink = RecordingSink()

    report_all_groupings(
        [_executed("only")],
        {"only": [JACOBI]},
        _groupings(_group(GAUSSIAN, ids), _group(None, ids)),
        {},
        sink,
    )

    assert sink.reported == []


def test_all_cache_hit_batch_touches_no_destination(entry_ids: tuple[str, ...]) -> None:
    sink = RecordingSink()

    report_all_groupings(
        [_cache_hit(e) for e in entry_ids],
        {},
        _groupings(_group(GAUSSIAN, entry_ids), _group(None, entry_ids)),
        {},
        sink,
    )

    assert sink.reported == []


def test_members_with_different_preconditioner_lists_do_not_raise() -> None:
    """A name present in only some members still gets a sharing policy (union of specs)."""
    ids = tuple(f"s{i}" for i in range(MIN_GROUP_SIZE))
    sink = RecordingSink()
    specs = {entry_id: [JACOBI] for entry_id in ids}
    # the last member also ran a second preconditioner the first member lacks
    extra = StandardPreconditionerConfig(type=PreconditionerType.ILU)
    specs[ids[-1]] = [JACOBI, extra]
    outcomes = [_executed(e) for e in ids]
    last = outcomes[-1].payload
    assert last is not None
    last.results[extra.name] = _cg_result(extra.name)

    report_all_groupings(
        outcomes, specs, _groupings(_group(GAUSSIAN, ids), _group(None, ids)), {}, sink
    )

    assert set(sink.reported[0][1]) == {JACOBI.name, extra.name}


def test_aggregated_outcome_carries_each_systems_time_to_solution(
    entry_ids: tuple[str, ...],
) -> None:
    sink = RecordingSink()

    report_all_groupings(
        [_executed(e) for e in entry_ids],
        {e: [JACOBI] for e in entry_ids},
        _groupings(_group(GAUSSIAN, entry_ids), _group(None, entry_ids)),
        {},
        sink,
    )

    per_system = sink.outcomes[0].time_to_solution_by_system[JACOBI.name]
    assert list(per_system) == list(entry_ids)  # config order
    assert set(per_system.values()) == {1.5}  # setup 1.0 + solve 0.5
