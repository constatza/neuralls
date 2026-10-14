"""Tests for the metrics table and its three report levels (``domain.solver.metrics_table``).

Pure reduction over hand-built ``CGComparisonResult`` fixtures — nothing here
runs a solve.

Fixture layout: two datasets (``ds-a``, ``ds-b``) times two recipes (``gauss``,
``sine``), two systems per cell, one preconditioner ``jacobi`` whose solve
time is ``base + 0.2 * j`` for the j-th system of a cell (iterations likewise
``base_it + 2 * j``), so every mean and std below is hand-computable.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.solver.metrics_table import (
    FineLevel,
    GlobalLevel,
    MediumLevel,
    SystemTag,
    build_metrics_frame,
    report_level,
    time_to_solution_by_system,
)
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    FineLabel,
    GlobalLabel,
    MediumLabel,
    SetupSharingPolicy,
    StageCost,
)
from neuralls.shared.types import CostProvenance, DatasetId, MatrixId

NOT_SHARED = SetupSharingPolicy(share_setup=False, share_generation=False)
SHARED = SetupSharingPolicy(share_setup=True, share_generation=True)
SHARING = {"jacobi": NOT_SHARED}

# (dataset, recipe) -> base solve seconds; the two systems of a cell are base and base + 0.2.
CELL_BASE_SOLVE = {
    ("ds-a", "gauss"): 1.0,
    ("ds-a", "sine"): 3.0,
    ("ds-b", "gauss"): 5.0,
    ("ds-b", "sine"): 7.0,
}


def _result(
    *,
    iterations: int = 5,
    setup_cost: float = 1.0,
    solve_time_seconds: float | None = 1.0,
    generation_cost: StageCost | None = None,
) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=iterations,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 1e-9],
        residual_history_abs=[1.0, 1e-9],
        preconditioner="jacobi",
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        generation_cost=generation_cost,
        setup_cost=setup_cost,
        solve_time_seconds=solve_time_seconds,
    )


def _tag(dataset: str, recipe: str, ordinal: int) -> SystemTag:
    return SystemTag(
        dataset=DatasetId(dataset),
        recipe=recipe,
        matrix_id=MatrixId(0),
        rhs_ordinal=ordinal,
        system_id=f"{dataset}/{recipe}/{ordinal}",
    )


@pytest.fixture
def grid_results() -> dict[SystemTag, dict[str, CGComparisonResult]]:
    return {
        _tag(dataset, recipe, j): {
            "jacobi": _result(
                iterations=int(base) + 2 * j, solve_time_seconds=base + 0.2 * j, setup_cost=0.5
            )
        }
        for (dataset, recipe), base in CELL_BASE_SOLVE.items()
        for j in range(2)
    }


def test_fine_level_reports_each_dataset_recipe_cell(grid_results) -> None:
    report = report_level(build_metrics_frame(grid_results), FineLevel(), SHARING)

    assert set(report) == {FineLabel(DatasetId(d), r) for d, r in CELL_BASE_SOLVE}
    cell = report[FineLabel(DatasetId("ds-a"), "sine")]["jacobi"]
    assert cell.n_systems == 2
    assert cell.avg_solve_time_per_system_seconds == pytest.approx(3.1)
    # sample std of {3.0, 3.2} is 0.2 / sqrt(2) * ... = |diff| / sqrt(2)
    assert cell.solve_time_std_seconds == pytest.approx(0.2 / 2**0.5)
    assert cell.total_iterations == 3 + 5
    assert cell.iterations_std == pytest.approx(2 / 2**0.5)


def test_medium_level_pools_recipes_within_a_dataset(grid_results) -> None:
    report = report_level(build_metrics_frame(grid_results), MediumLevel(), SHARING)

    label = MediumLabel(DatasetId("ds-b"), mixture_of=("gauss", "sine"))
    assert set(report) == {MediumLabel(DatasetId("ds-a"), ("gauss", "sine")), label}
    pooled = report[label]["jacobi"]
    values = [5.0, 5.2, 7.0, 7.2]
    mean = sum(values) / 4
    sample_var = sum((v - mean) ** 2 for v in values) / 3
    assert pooled.n_systems == 4
    assert pooled.avg_solve_time_per_system_seconds == pytest.approx(mean)
    assert pooled.solve_time_std_seconds == pytest.approx(sample_var**0.5)


def test_global_level_pools_every_system(grid_results) -> None:
    report = report_level(build_metrics_frame(grid_results), GlobalLevel(), SHARING)

    assert list(report) == [GlobalLabel(mixture_of=("gauss", "sine"))]
    pooled = report[GlobalLabel(("gauss", "sine"))]["jacobi"]
    values = [b + 0.2 * j for b in CELL_BASE_SOLVE.values() for j in range(2)]
    mean = sum(values) / 8
    assert pooled.n_systems == 8
    assert pooled.avg_solve_time_per_system_seconds == pytest.approx(mean)
    assert pooled.solve_time_std_seconds == pytest.approx(
        (sum((v - mean) ** 2 for v in values) / 7) ** 0.5
    )


def test_a_single_system_group_has_no_spread() -> None:
    results = {_tag("ds-a", "gauss", 0): {"jacobi": _result(solve_time_seconds=2.0)}}

    projected = report_level(build_metrics_frame(results), GlobalLevel(), SHARING)[
        GlobalLabel(("gauss",))
    ]["jacobi"]

    assert projected.n_systems == 1
    assert projected.solve_time_std_seconds is None
    assert projected.iterations_std is None


def test_a_missing_solve_time_is_skipped_not_counted_as_zero() -> None:
    results = {
        _tag("ds-a", "gauss", 0): {"jacobi": _result(solve_time_seconds=2.0)},
        _tag("ds-a", "gauss", 1): {"jacobi": _result(solve_time_seconds=None)},
        _tag("ds-a", "gauss", 2): {"jacobi": _result(solve_time_seconds=4.0)},
    }

    projected = report_level(build_metrics_frame(results), GlobalLevel(), SHARING)[
        GlobalLabel(("gauss",))
    ]["jacobi"]

    assert projected.avg_solve_time_per_system_seconds == pytest.approx(3.0)
    assert projected.solve_time_std_seconds == pytest.approx(2**0.5)


def test_generation_average_divides_by_available_results_only() -> None:
    historical = StageCost(
        wall_time_seconds=6.0, peak_memory_bytes=None, provenance=CostProvenance.HISTORICAL
    )
    unavailable = StageCost(
        wall_time_seconds=0.0, peak_memory_bytes=None, provenance=CostProvenance.UNAVAILABLE
    )
    results = {
        _tag("ds-a", "gauss", 0): {"jacobi": _result(generation_cost=historical)},
        _tag("ds-a", "gauss", 1): {"jacobi": _result(generation_cost=unavailable)},
    }
    frame = build_metrics_frame(results)

    projected = report_level(frame, GlobalLevel(), SHARING)[GlobalLabel(("gauss",))]["jacobi"]

    assert frame["generation_s"].to_list() == [6.0, None]
    assert projected.avg_generation_cost_seconds == pytest.approx(6.0)


def test_all_unavailable_generation_is_none() -> None:
    unavailable = StageCost(
        wall_time_seconds=0.0, peak_memory_bytes=None, provenance=CostProvenance.UNAVAILABLE
    )
    results = {_tag("ds-a", "gauss", 0): {"jacobi": _result(generation_cost=unavailable)}}

    projected = report_level(build_metrics_frame(results), GlobalLevel(), SHARING)[
        GlobalLabel(("gauss",))
    ]["jacobi"]

    assert projected.generation_cost_total_seconds is None


def test_sharing_policy_decides_how_setup_scales_with_n() -> None:
    results = {_tag("ds-a", "gauss", j): {"jacobi": _result(setup_cost=7.0)} for j in range(3)}
    frame = build_metrics_frame(results)
    label = GlobalLabel(("gauss",))

    shared = report_level(frame, GlobalLevel(), {"jacobi": SHARED})[label]["jacobi"]
    per_system = report_level(frame, GlobalLevel(), {"jacobi": NOT_SHARED})[label]["jacobi"]

    assert shared.setup_cost_total_seconds == pytest.approx(21.0)
    assert shared.projected_total_seconds(10) == pytest.approx(7.0 + 10 * 1.0)
    assert per_system.projected_total_seconds(10) == pytest.approx(70.0 + 10 * 1.0)


def test_missing_sharing_policy_raises_key_error(grid_results) -> None:
    with pytest.raises(KeyError):
        report_level(build_metrics_frame(grid_results), GlobalLevel(), {})


def test_time_to_solution_by_system_omits_incomparable_systems() -> None:
    unavailable = StageCost(
        wall_time_seconds=0.0, peak_memory_bytes=None, provenance=CostProvenance.UNAVAILABLE
    )
    results = {
        _tag("ds-a", "gauss", 0): {"jacobi": _result(setup_cost=1.0, solve_time_seconds=0.5)},
        _tag("ds-a", "gauss", 1): {"jacobi": _result(generation_cost=unavailable)},
    }

    by_label = time_to_solution_by_system(build_metrics_frame(results), GlobalLevel())

    assert by_label == {GlobalLabel(("gauss",)): {"jacobi": {"ds-a/gauss/0": 1.5}}}


def test_frame_has_one_row_per_system_and_preconditioner(grid_results) -> None:
    frame = build_metrics_frame(grid_results)

    assert frame.height == 8
    assert frame.columns == [
        "dataset", "recipe", "matrix_id", "rhs_ordinal", "system_id", "preconditioner",
        "setup_s", "generation_s", "solve_s", "iterations", "time_to_solution_s",
        "peak_memory_bytes",
    ]  # fmt: skip
