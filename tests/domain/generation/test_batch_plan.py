"""Row-budget plan: totals must agree with the resolved strategy counts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.generation.batch_plan import BindingAllocation, plan_batches
from neuralls.domain.generation.helpers import resolve_strategy_counts
from neuralls.domain.generation.orchestration import _resolve_binding_strategy_counts
from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec

_NUM_MATRICES = 3


@pytest.fixture
def multi_matrix_spec() -> DatasetSpec:
    """Two generated strategies over several matrices, with a residual window override."""
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={"gaussian_forward": 6, "gaussian_residuals": 6},
            seed=1234,
            shuffle=False,
            strategy_overrides={"gaussian_residuals": {"stop": 3, "start": 1}},
        ),
        normalize="matrix",
    )


def test_plan_totals_match_strategy_counts(
    multi_matrix_spec: DatasetSpec,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    bindings = make_bindings(_NUM_MATRICES, 1)
    allocation = _resolve_binding_strategy_counts(
        bindings=bindings,
        spec=multi_matrix_spec,
        num_matrix_samples=_NUM_MATRICES,
    )
    plan = plan_batches(allocation)

    budget = resolve_strategy_counts(
        multi_matrix_spec.mixture.counts,
        multi_matrix_spec.mixture.mix,
        multi_matrix_spec.mixture.total,
    )
    assert plan.total_rows == sum(budget.values())
    assert dict(plan.strategy_totals) == budget
    assert sum(binding.total_rows for binding in plan.bindings) == plan.total_rows


@pytest.fixture
def zero_count_allocation() -> BindingAllocation:
    """One binding that asks for one generated strategy and one archive strategy with no rows."""
    return BindingAllocation(
        counts=({"gaussian_forward": 4, "rhs_archive": 0},),
        file_indices=({},),
    )


def test_plan_omits_zero_count_strategies(zero_count_allocation: BindingAllocation) -> None:
    """A zero-count strategy is never run, so it must not reach the plan."""
    plan = plan_batches(zero_count_allocation)
    assert [strategy.name for strategy in plan.bindings[0].strategies] == ["gaussian_forward"]
    assert plan.total_rows == 4
