"""Tests for plot orchestration and plot-path serialization (no figure is inspected)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.platform.reporting import plots as reporting_plots
from neuralls.platform.reporting.figure_text import (
    DID_NOT_CONVERGE_MARK,
    EditorialTitle,
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


@pytest.fixture
def comparison_paths(tmp_path: Path) -> ComparisonPaths:
    """Comparison paths rooted in pytest-managed temporary storage."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True)
    return ComparisonPaths(
        matrix=tmp_path / "matrix.npy",
        rhs=tmp_path / "rhs.npy",
        output=tmp_path,
        figures=figures_dir,
    )


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


# ---------------------------------------------------------------------------
# _generate_comparison_plots — includes iterations_barplot
# ---------------------------------------------------------------------------


def test_generate_comparison_plots_includes_iterations_barplot(
    comparison_paths: ComparisonPaths,
    two_result_entries: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
) -> None:
    """_generate_comparison_plots returns PlotPaths with iterations_barplot set.

    Plotting functions are patched to avoid file I/O while still exercising the
    orchestration logic that populates PlotPaths.
    """
    legend = legend_of(*two_result_entries)
    title = EditorialTitle("demo")

    with (
        patch(
            "neuralls.composition.comparison._plots.plot_convergence_comparison"
        ) as convergence_plot,
        patch(
            "neuralls.composition.comparison._plots.plot_error_convergence_comparison"
        ) as error_plot,
        patch("neuralls.composition.comparison._plots.plot_method_bars") as bars_plot,
    ):
        result = _generate_comparison_plots(two_result_entries, comparison_paths, legend, title)

    assert result.time_breakdown_barplot is None  # neither entry has setup/solve times
    assert "condition_numbers" not in result.to_mapping()
    assert convergence_plot.call_args.kwargs["title"] == title
    assert convergence_plot.call_args.args[0] is two_result_entries
    assert convergence_plot.call_args.kwargs["legend"] == legend
    assert error_plot.call_args.args[0] is two_result_entries
    assert error_plot.call_args.kwargs["legend"] == legend
    assert bars_plot.call_args.kwargs["title"] == title


def test_generate_comparison_plots_includes_time_breakdown_barplot_when_measured(
    comparison_paths: ComparisonPaths,
    two_result_entries_with_times: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
) -> None:
    """_generate_comparison_plots populates time_breakdown_barplot when times are measured."""
    with (
        patch("neuralls.composition.comparison._plots.plot_convergence_comparison"),
        patch("neuralls.composition.comparison._plots.plot_error_convergence_comparison"),
        patch("neuralls.composition.comparison._plots.plot_method_bars"),
    ):
        result = _generate_comparison_plots(
            two_result_entries_with_times,
            comparison_paths,
            legend_of(*two_result_entries_with_times),
            EditorialTitle("demo"),
        )
        assert result.time_breakdown_barplot is not None


def test_generate_comparison_plots_marks_only_the_unconverged_result(
    comparison_paths: ComparisonPaths,
    two_result_entries: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
) -> None:
    """The did-not-converge mark reaches the plots through the legend, never the result key."""
    failed, kept = list(two_result_entries)
    results = {**two_result_entries, failed: replace(two_result_entries[failed], converged=False)}
    legend = legend_of(*results)

    with (
        patch(
            "neuralls.composition.comparison._plots.plot_convergence_comparison"
        ) as convergence_plot,
        patch("neuralls.composition.comparison._plots.plot_error_convergence_comparison"),
        patch("neuralls.composition.comparison._plots.plot_method_bars"),
    ):
        _generate_comparison_plots(results, comparison_paths, legend, EditorialTitle("demo"))

    drawn = convergence_plot.call_args.kwargs["legend"]
    assert drawn.label(kept) == legend.label(kept)
    assert drawn.label(failed) == f"{legend.label(failed)} {DID_NOT_CONVERGE_MARK}"
