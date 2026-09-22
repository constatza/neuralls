"""Resilience of the CG comparison runner against reference-solve failures."""

from __future__ import annotations

from unittest.mock import patch

import torch

from neuralls.domain.solver import comparison
from neuralls.domain.solver.comparison import run_cg_comparison


def test_reference_failure_drops_exact_metrics_but_falls_back_to_gm_bound(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """A failing reference solve leaves exact-error metrics unset and still solves,

    falling back to the Golub-Meurant bound for the energy-error signal instead of
    losing it entirely.
    """
    with patch.object(comparison, "reference_solution", side_effect=RuntimeError("too ill")):
        results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners, rtol=1e-8)

    assert all(result.converged for result in results.values())
    assert all(result.exact_error is None for result in results.values())
    assert all(result.error_history_a_rel is None for result in results.values())
    assert all(result.error_bound_a_rel is not None for result in results.values())


def test_energy_error_fields_are_mutually_exclusive(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """Exactly one of error_history_a_rel / error_bound_a_rel is set, never both."""
    results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners, rtol=1e-8)

    for result in results.values():
        assert (result.error_history_a_rel is None) != (result.error_bound_a_rel is None)
