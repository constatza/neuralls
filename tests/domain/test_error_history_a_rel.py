"""Tests for ``CGComparisonResult.error_history_a_rel`` produced by a real comparison run."""

from __future__ import annotations

import itertools

import pytest
import torch

from neuralls.domain.solver import comparison
from neuralls.domain.solver.comparison import run_cg_comparison
from neuralls.domain.solver.error_metrics import REFERENCE_PRECISION_MARGIN, reference_solution


def test_history_starts_at_one_and_is_monotone_for_pcg(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners, rtol=1e-10)
    for result in results.values():
        history = result.error_history_a_rel
        assert history is not None
        assert history[0] == pytest.approx(1.0)
        assert len(history) == len(result.residual_history_rel)
        assert all(b <= a * (1 + 1e-9) for a, b in itertools.pairwise(history))


def test_timed_solve_never_receives_the_exact_solution(
    monkeypatch: pytest.MonkeyPatch,
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """The error history needs ``x_exact``; tracking it per iteration must not be billed to the solve."""
    real_pcg = comparison.pcg
    exact_given: list[bool] = []

    def recording_pcg(*args: object, **kwargs: object) -> object:
        exact_given.append(kwargs.get("x_exact") is not None)
        return real_pcg(*args, **kwargs)

    monkeypatch.setattr(comparison, "pcg", recording_pcg)

    results = run_cg_comparison(
        spd_matrix, rhs, preconditioners={"only": preconditioners["jacobi"]}, rtol=1e-10
    )

    assert exact_given == [False, True]
    assert results["only"].error_history_a_rel is not None


def test_reference_solution_beats_rtol_by_margin(
    spd_matrix: torch.Tensor, rhs: torch.Tensor, known_solution: torch.Tensor
) -> None:
    x = reference_solution(spd_matrix, rhs, rtol=1e-6)
    relative_residual = torch.linalg.vector_norm(rhs - spd_matrix @ x) / torch.linalg.vector_norm(
        rhs
    )
    assert float(relative_residual) <= 1e-6 * REFERENCE_PRECISION_MARGIN
    torch.testing.assert_close(x, known_solution)
