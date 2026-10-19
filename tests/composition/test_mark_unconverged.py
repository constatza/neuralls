"""``mark_unconverged``: a capped run is labelled so per-system plots never show it as cheap."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neuralls.composition.comparison._plots import DID_NOT_CONVERGE_MARK, mark_unconverged
from neuralls.domain.solver.models.cg_result import CGComparisonResult


@pytest.fixture
def converged_result() -> CGComparisonResult:
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=4,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 1e-9],
        residual_history_abs=[1.0, 1e-9],
        preconditioner="amg",
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=1.0,
        solve_time_seconds=0.5,
    )


def test_only_the_unconverged_run_is_marked(converged_result) -> None:
    results = {"amg": converged_result, "jacobi": replace(converged_result, converged=False)}

    marked = mark_unconverged(results, {"amg": "AMG", "jacobi": "Jacobi"})

    assert marked == {"amg": "AMG", "jacobi": f"Jacobi {DID_NOT_CONVERGE_MARK}"}
