"""Tests for ``CGComparisonResult.error_history_a_rel`` produced by a real comparison run."""

from __future__ import annotations

import itertools

import pytest
import torch

from neuralls.domain.solver.comparison import run_cg_comparison
from neuralls.domain.solver.error_metrics import reference_solution


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


def test_reference_solution_beats_rtol_by_margin(
    spd_matrix: torch.Tensor, rhs: torch.Tensor, known_solution: torch.Tensor
) -> None:
    x = reference_solution(spd_matrix, rhs, rtol=1e-6)
    relative_residual = torch.linalg.vector_norm(rhs - spd_matrix @ x) / torch.linalg.vector_norm(
        rhs
    )
    assert float(relative_residual) <= 1e-6 * 1e-4
    torch.testing.assert_close(x, known_solution)
