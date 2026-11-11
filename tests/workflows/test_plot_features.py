"""Tests for plot orchestration and plot-path serialization (no figure is inspected)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest

from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.platform.reporting import convergence_plots as reporting_plots
from neuralls.platform.reporting.convergence_plots import (
    plot_convergence_comparison,
)
from neuralls.platform.reporting.figure_text import (
    Legend,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_result_entries() -> dict[str, CGComparisonResult]:
    """Two minimal CGComparisonResult entries for plot tests.

    Returns:
        A dict with two entries: "identity" and "jacobi", each populated with
        short residual histories suitable for plot testing.
    """

    def _make(name: str, iters: int) -> CGComparisonResult:
        history = [1.0 / (k + 1) for k in range(iters)]
        return CGComparisonResult(
            x=np.zeros(2),
            converged=True,
            iterations=iters,
            residual=history[-1],
            residual_abs=history[-1],
            residual_history_rel=history,
            residual_history_abs=history,
            preconditioner=name,
            initial_guess=np.zeros(2),
            exact_error=None,
            rhs_norm=1.0,
            breakdown=False,
        )

    return {
        "identity": _make("identity", 5),
        "jacobi": _make("jacobi", 3),
    }


# ---------------------------------------------------------------------------
# plot_metric_comparison — horizontal mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# plot_time_breakdown_barplot
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# plot_convergence_comparison — new parameters
# ---------------------------------------------------------------------------


def test_plot_convergence_comparison_without_a_legend_entry_raises(
    two_result_entries: dict[str, CGComparisonResult], legend_of: Callable[..., Legend]
) -> None:
    """A result with no presentation never falls back to its key."""
    first = next(iter(two_result_entries))

    with (
        patch("neuralls.platform.reporting.convergence_plots.plt.close"),
        pytest.raises(MissingMethodPresentationError) as raised,
    ):
        plot_convergence_comparison(two_result_entries, legend=legend_of(first))

    assert raised.value.method_id in set(two_result_entries) - {first}
    reporting_plots.plt.close("all")
