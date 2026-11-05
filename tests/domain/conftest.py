"""Fixtures for domain-level solver tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import torch
from torchalg.models.result import SolverResult
from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner

from neuralls.domain.solver.models.benchmark_events import SolveConverged
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost import Known, NotApplicable, OfflineStages
from neuralls.domain.solver.reporting.observations import MethodFrames, observations_of_methods
from neuralls.shared.types import CostProvenance, PreconditionerName
from tests import observation_support
from tests.observation_support import MethodEvents


@pytest.fixture
def spd_matrix() -> torch.Tensor:
    """Seeded well-conditioned SPD matrix (n=20)."""
    generator = torch.Generator().manual_seed(0)
    m = torch.randn(20, 20, dtype=torch.float64, generator=generator)
    return m @ m.T + 20 * torch.eye(20, dtype=torch.float64)


@pytest.fixture
def known_solution() -> torch.Tensor:
    """Seeded ground-truth solution x*."""
    generator = torch.Generator().manual_seed(1)
    return torch.randn(20, dtype=torch.float64, generator=generator)


@pytest.fixture
def rhs(spd_matrix: torch.Tensor, known_solution: torch.Tensor) -> torch.Tensor:
    """Right-hand side generated as b = A x*."""
    return spd_matrix @ known_solution


@pytest.fixture
def preconditioners(spd_matrix: torch.Tensor) -> dict[str, object]:
    """Identity and Jacobi preconditioners for the SPD fixture.

    Keyed "identity", matching `run_cg_comparison`'s own auto-injected
    baseline key — providing it explicitly here means no second, redundant
    baseline solve gets silently added on top of these two.
    """
    return {"identity": Identity(), "jacobi": JacobiPreconditioner().setup(spd_matrix)}


@pytest.fixture
def perturbed_solution(known_solution: torch.Tensor) -> torch.Tensor:
    """A solution offset from ``known_solution`` by a fixed, known amount."""
    return known_solution + 0.1


@pytest.fixture
def zero_reference_solution() -> torch.Tensor:
    """A zero reference solution, exercising ``relative_exact_error``'s ~0-norm branch."""
    return torch.zeros(20, dtype=torch.float64)


@pytest.fixture
def decreasing_error_history() -> list[float]:
    """A monotonically decreasing error/bound history with a nonzero first entry."""
    return [4.0, 2.0, 1.0, 0.5]


@pytest.fixture
def zero_first_error_history() -> list[float]:
    """An error/bound history whose first entry is zero (undefined normalization)."""
    return [0.0, 1.0, 2.0]


@pytest.fixture
def solver_result_with_exact_energy_history() -> SolverResult:
    """A SolverResult carrying the exact ``error_history_a_norm`` (x_exact was supplied)."""
    return SolverResult(
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_norm=(4.0, 2.0, 1.0, 0.5),
        energy_decrements=(1.0, 0.5, 0.25),
    )


@pytest.fixture
def solver_result_with_decrements_only() -> SolverResult:
    """A SolverResult with no exact history (no x_exact) but nonempty energy_decrements."""
    return SolverResult(
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_norm=None,
        energy_decrements=(1.0, 0.5, 0.25),
    )


@pytest.fixture
def comparison_result_with_cost_data() -> CGComparisonResult:
    """A converged CGComparisonResult with setup/solve time and memory measured."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        preconditioner="jacobi",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=2.0,
        setup_peak_memory_bytes=1000,
        solve_time_seconds=1.0,
        solve_peak_memory_bytes=2000,
    )


@pytest.fixture
def comparison_result_with_offline_cost() -> CGComparisonResult:
    """A converged CGComparisonResult carrying a historical generation
    cost and a freshly-measured setup cost."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        preconditioner="pod2g",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        offline_cost=OfflineStages(
            generation=Known(
                wall_time_seconds=5.0,
                peak_memory_bytes=500,
                provenance=CostProvenance.HISTORICAL,
            ),
            training=NotApplicable(),
        ),
        setup_cost=2.0,
        setup_peak_memory_bytes=1000,
        solve_time_seconds=1.0,
        solve_peak_memory_bytes=2000,
    )


@pytest.fixture
def comparison_result_without_cost_data() -> CGComparisonResult:
    """A converged CGComparisonResult with no cost measurements attached."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        preconditioner="jacobi",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
    )


@pytest.fixture
def comparison_result_with_zero_iterations() -> CGComparisonResult:
    """A CGComparisonResult where the "identity" baseline converged immediately (0 iterations)."""
    return CGComparisonResult(
        x=np.zeros(10),
        converged=True,
        iterations=0,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=[1e-8],
        residual_history_abs=[1e-8],
        preconditioner="identity",
        initial_guess=np.zeros(10),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        solve_time_seconds=0.0,
        solve_peak_memory_bytes=0,
    )


@pytest.fixture
def solver_result_with_no_energy_data() -> SolverResult:
    """A SolverResult with neither the exact history nor energy_decrements populated."""
    return SolverResult(
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_norm=None,
        energy_decrements=None,
    )


@pytest.fixture
def unbalanced_events() -> Mapping[PreconditionerName, MethodEvents]:
    """Domain events of three methods over an unbalanced workload with every outcome."""
    return observation_support.unbalanced_events()


@pytest.fixture
def unbalanced_frames(
    unbalanced_events: Mapping[PreconditionerName, MethodEvents],
) -> Mapping[PreconditionerName, MethodFrames]:
    """The stored frames of the three methods."""
    return {method: events.frames() for method, events in unbalanced_events.items()}


@pytest.fixture
def unbalanced_observations(
    unbalanced_frames: Mapping[PreconditionerName, MethodFrames],
) -> pl.DataFrame:
    """The observation view of all three methods."""
    return observations_of_methods(unbalanced_frames)


@pytest.fixture
def two_recipe_observations(unbalanced_observations: pl.DataFrame) -> pl.DataFrame:
    """The observation view of all three methods, matrices 3 and 4 drawn from a second recipe."""
    return observation_support.with_second_recipe(unbalanced_observations)


@pytest.fixture
def two_method_observations(unbalanced_observations: pl.DataFrame) -> pl.DataFrame:
    """The observation view of ``amg`` and ``jacobi`` only, the pair that share systems."""
    return unbalanced_observations.filter(
        pl.col("method").is_in([observation_support.AMG, observation_support.JACOBI])
    )


@pytest.fixture
def drifted_observations(two_method_observations: pl.DataFrame) -> pl.DataFrame:
    """Observations where one converged ``amg`` system's true residual is 20 times its threshold."""
    return two_method_observations.with_columns(
        true_residual_rel=pl.when(
            (pl.col("system_id") == "m1-0") & (pl.col("method") == observation_support.AMG)
        )
        .then(20 * observation_support.STOPPING_THRESHOLD_REL)
        .otherwise(pl.col("true_residual_rel"))
    )


@pytest.fixture
def zero_iteration_events(
    unbalanced_events: Mapping[PreconditionerName, MethodEvents],
) -> Mapping[PreconditionerName, MethodEvents]:
    """The unbalanced events where ``amg`` converges system ``m1-0`` at the initial guess.

    ``amg`` needs 0 iterations there and ``jacobi`` 10, so the best iteration count of ``m1-0``
    is 0 and a ratio to it is undefined.
    """
    amg = unbalanced_events[observation_support.AMG]
    attempts = tuple(
        replace(attempt, outcome=SolveConverged(0, 1e-9))
        if attempt.system_id == "m1-0"
        else attempt
        for attempt in amg.solve_attempts
    )
    return {**unbalanced_events, amg.method: replace(amg, solve_attempts=attempts)}


@pytest.fixture
def zero_iteration_observations(
    zero_iteration_events: Mapping[PreconditionerName, MethodEvents],
) -> pl.DataFrame:
    """The observation view of ``zero_iteration_events``."""
    return observations_of_methods(
        {method: events.frames() for method, events in zero_iteration_events.items()}
    )


@pytest.fixture
def zero_solve_time_observations(two_method_observations: pl.DataFrame) -> pl.DataFrame:
    """Two methods where ``amg`` solves every system of matrix 1 in zero seconds."""
    return two_method_observations.with_columns(
        solve_s=pl.when((pl.col("method") == observation_support.AMG) & (pl.col("matrix_id") == 1))
        .then(0.0)
        .otherwise(pl.col("solve_s"))
    )


@pytest.fixture
def setup_heavy_fast_solver_observations(two_method_observations: pl.DataFrame) -> pl.DataFrame:
    """Two methods where ``amg`` solves ten times faster than in the base workload.

    ``amg`` then averages 0.25 s per common system against 0.65 s for ``jacobi``, while its
    matrix setups (6 s and 12 s) stay far above ``jacobi``'s (1 s and 2 s).
    """
    return two_method_observations.with_columns(
        solve_s=pl.when(pl.col("method") == observation_support.AMG)
        .then(pl.col("solve_s") / 10)
        .otherwise(pl.col("solve_s"))
    )
