"""Tests for plot orchestration and plot-path serialization (no figure is inspected)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.platform.reporting import plots as reporting_plots
from neuralls.platform.reporting.figure_text import (
    Legend,
)
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESIDUAL_HISTORY: list[float] = [1.0, 0.5, 0.1, 0.01, 1e-6]
RESIDUAL_HISTORY_ABS: list[float] = [1.0, 0.5, 0.1, 0.01, 1e-6]


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


@pytest.fixture
def two_result_entries_with_times() -> dict[str, CGComparisonResult]:
    """Two minimal CGComparisonResult entries with setup/solve times set.

    Returns:
        A dict with two entries: "identity" and "jacobi", each carrying
        distinct ``setup_time_seconds``/``solve_time_seconds`` for
        time-breakdown plot tests.
    """

    def _make(name: str, setup: float, solve: float) -> CGComparisonResult:
        return CGComparisonResult(
            x=np.zeros(2),
            converged=True,
            iterations=3,
            residual=1e-6,
            residual_abs=1e-6,
            residual_history_rel=[1.0, 0.1, 1e-6],
            residual_history_abs=[1.0, 0.1, 1e-6],
            preconditioner=name,
            initial_guess=np.zeros(2),
            exact_error=None,
            rhs_norm=1.0,
            breakdown=False,
            setup_cost=setup,
            solve_time_seconds=solve,
        )

    return {
        "identity": _make("identity", setup=0.5, solve=0.1),
        "jacobi": _make("jacobi", setup=0.2, solve=2.0),
    }


# ---------------------------------------------------------------------------
# PlotPaths serialization
# ---------------------------------------------------------------------------


def test_plot_paths_iterations_barplot_round_trips_via_mapping() -> None:
    """PlotPaths with iterations_barplot survives a to_mapping/from_mapping round-trip.

    Verifies that the field is present in the serialised mapping and is
    reconstructed correctly by from_mapping.
    """
    expected_path = Path("out/iters.png")
    plot_paths = PlotPaths(iterations_barplot=expected_path)

    mapping = plot_paths.to_mapping()

    assert "iterations_barplot" in mapping
    assert mapping["iterations_barplot"] == expected_path

    reconstructed = PlotPaths.from_mapping(mapping)
    assert reconstructed.iterations_barplot == expected_path


def test_plot_paths_none_iterations_barplot_not_in_mapping() -> None:
    """PlotPaths without iterations_barplot omits key from to_mapping result.

    Ensures that None-valued fields are not emitted to the sparse mapping dict.
    """
    plot_paths = PlotPaths()
    mapping = plot_paths.to_mapping()
    assert "iterations_barplot" not in mapping


def test_plot_paths_time_breakdown_barplot_round_trips_via_mapping() -> None:
    """PlotPaths with time_breakdown_barplot survives a to_mapping/from_mapping round-trip."""
    expected_path = Path("out/time_breakdown.png")
    plot_paths = PlotPaths(time_breakdown_barplot=expected_path)

    mapping = plot_paths.to_mapping()

    assert "time_breakdown_barplot" in mapping
    assert mapping["time_breakdown_barplot"] == expected_path

    reconstructed = PlotPaths.from_mapping(mapping)
    assert reconstructed.time_breakdown_barplot == expected_path


def test_plot_paths_none_time_breakdown_barplot_not_in_mapping() -> None:
    """PlotPaths without time_breakdown_barplot omits key from to_mapping result."""
    plot_paths = PlotPaths()
    mapping = plot_paths.to_mapping()
    assert "time_breakdown_barplot" not in mapping


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
        patch("neuralls.platform.reporting.plots.plt.close"),
        pytest.raises(MissingMethodPresentationError) as raised,
    ):
        plot_convergence_comparison(two_result_entries, legend=legend_of(first))

    assert raised.value.method_id in set(two_result_entries) - {first}
    reporting_plots.plt.close("all")


# ---------------------------------------------------------------------------
# plot_condition_numbers — horizontal bars, title, tolerances
# ---------------------------------------------------------------------------
