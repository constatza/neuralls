"""Tests for recently added plot features in plotting.py, spectra.py, and compare.py."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.domain.analysis.spectra import plot_condition_numbers
from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost import StageCost
from neuralls.domain.solver.models.figure_scope import AllDatasetsScope
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
)
from neuralls.platform.reporting import plots as reporting_plots
from neuralls.platform.reporting.figure_text import (
    DID_NOT_CONVERGE_MARK,
    EditorialTitle,
    Legend,
    ScopeSubtitle,
)
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
    plot_metric_comparison,
    plot_time_breakdown_barplot,
    plot_work_precision,
)
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import CostProvenance

# TODO: Keep one real save per public plotter; mock the render/save boundary in
# behavioral tests that only verify sorting, labels, collision handling, or omission.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RTOL = 1e-6
ATOL = 1e-14
MAX_ITERATIONS = 100
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
def two_result_entries_with_identical_generation_time_different_provenance() -> dict[
    str, CGComparisonResult
]:
    """Two results sharing an identical ``generation_cost.wall_time_seconds`` but
    different provenance -- one MEASURED (a genuinely fast fit), one
    UNAVAILABLE (a silently-failed historical lookup that fell back to the
    same near-zero measured value). ``setup_cost`` is now always a plain
    measured float (no provenance), so generation cost is the only
    remaining stage that can carry UNAVAILABLE provenance. The direct
    regression fixture for that: these two must never render identically.
    """

    def _make(name: str, provenance: CostProvenance) -> CGComparisonResult:
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
            generation_cost=StageCost(
                wall_time_seconds=0.002, peak_memory_bytes=None, provenance=provenance
            ),
            setup_cost=0.1,
            solve_time_seconds=0.1,
        )

    return {
        "measured": _make("measured", CostProvenance.MEASURED),
        "unavailable": _make("unavailable", CostProvenance.UNAVAILABLE),
    }


@pytest.fixture
def simple_cond_numbers() -> dict[str, float]:
    """Simple condition number mapping for two preconditioners.

    Returns:
        Mapping from preconditioner name to condition number float.
    """
    return {"identity": 1.0, "jacobi": 5.0}


@pytest.fixture
def two_preconditioner_families() -> dict[str, PreconditionerType]:
    """Family key per name, matching ``two_preconditioners``' keys.

    Returns:
        A dict with two entries: "identity" and "jacobi", each mapped to its
        ``PreconditionerType``.
    """
    return {
        "identity": PreconditionerType.IDENTITY,
        "jacobi": PreconditionerType.JACOBI,
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


@pytest.fixture
def metric_labels() -> list[str]:
    """Short labels for a metric comparison chart.

    Returns:
        List of short label strings.
    """
    return ["1", "2", "3"]


@pytest.fixture
def metric_values() -> list[float]:
    """Metric values aligned with metric_labels.

    Returns:
        List of floats representing per-label metric values.
    """
    return [0.001, 0.005, 0.002]


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


def test_plot_metric_comparison_horizontal_saves_file(
    tmp_path: Path,
    metric_labels: list[str],
    metric_values: list[float],
) -> None:
    """plot_metric_comparison with horizontal=True writes a PNG to save_path.

    Args:
        tmp_path: Pytest temporary directory.
        metric_labels: Short bar labels from fixture.
        metric_values: Matching metric values from fixture.
    """
    save_path = tmp_path / "out.png"
    plot_metric_comparison(
        metric_labels,
        metric_values,
        metric_name="CG Iterations",
        save_path=save_path,
        horizontal=True,
    )
    assert save_path.exists()


def test_plot_metric_comparison_vertical_saves_file(
    tmp_path: Path,
    metric_labels: list[str],
    metric_values: list[float],
) -> None:
    """plot_metric_comparison with horizontal=False writes a PNG to save_path.

    Args:
        tmp_path: Pytest temporary directory.
        metric_labels: Short bar labels from fixture.
        metric_values: Matching metric values from fixture.
    """
    save_path = tmp_path / "out.png"
    plot_metric_comparison(
        metric_labels,
        metric_values,
        metric_name="CG Iterations",
        save_path=save_path,
        horizontal=False,
    )
    assert save_path.exists()


def test_plot_metric_comparison_with_title_saves_file(
    tmp_path: Path,
    metric_labels: list[str],
    metric_values: list[float],
) -> None:
    """plot_metric_comparison with a custom title writes a PNG to save_path.

    Args:
        tmp_path: Pytest temporary directory.
        metric_labels: Short bar labels from fixture.
        metric_values: Matching metric values from fixture.
    """
    save_path = tmp_path / "out.png"
    plot_metric_comparison(
        metric_labels,
        metric_values,
        metric_name="CG Iterations",
        title="My Title",
        save_path=save_path,
    )
    assert save_path.exists()


def test_plot_metric_comparison_horizontal_sorts_bars_descending(
    metric_labels: list[str],
) -> None:
    """Horizontal plot_metric_comparison orders bars by value, largest first (at top)."""
    unsorted_values = [0.002, 0.005, 0.001]  # label "2" is the largest
    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_metric_comparison(
            metric_labels, unsorted_values, metric_name="Metric", horizontal=True
        )
        fig = reporting_plots.plt.gcf()
        ax = fig.axes[0]
        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    reporting_plots.plt.close(fig)

    assert yticklabels == ["2", "1", "3"]


def test_plot_metric_comparison_vertical_sorts_descending_and_rotates_labels(
    metric_labels: list[str],
) -> None:
    """Vertical plot_metric_comparison orders bars by value (desc) and tilts x-tick labels."""
    unsorted_values = [0.002, 0.005, 0.001]  # label "2" is the largest
    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_metric_comparison(
            metric_labels, unsorted_values, metric_name="Metric", horizontal=False
        )
        fig = reporting_plots.plt.gcf()
        ax = fig.axes[0]
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        rotation = ax.get_xticklabels()[0].get_rotation()
    reporting_plots.plt.close(fig)

    assert xticklabels == ["2", "1", "3"]
    assert rotation == 45


# ---------------------------------------------------------------------------
# plot_time_breakdown_barplot
# ---------------------------------------------------------------------------


def test_plot_time_breakdown_barplot_saves_file_when_times_present(
    tmp_path: Path,
    two_result_entries_with_times: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
) -> None:
    """plot_time_breakdown_barplot writes a PNG when results carry setup/solve times.

    Args:
        tmp_path: Pytest temporary directory.
        two_result_entries_with_times: Two-entry result dict with times set.
    """
    save_path = tmp_path / "breakdown.png"
    plot_time_breakdown_barplot(
        two_result_entries_with_times,
        legend_of(*two_result_entries_with_times),
        save_path=save_path,
    )
    assert save_path.exists()


def test_plot_time_breakdown_barplot_skips_methods_missing_both_times(
    tmp_path: Path,
    two_result_entries: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
) -> None:
    """plot_time_breakdown_barplot still writes a (empty) chart when no times are measured.

    Args:
        tmp_path: Pytest temporary directory.
        two_result_entries: Two-entry result dict fixture with no times set.
    """
    save_path = tmp_path / "breakdown.png"
    plot_time_breakdown_barplot(
        two_result_entries, legend_of(*two_result_entries), save_path=save_path
    )
    assert save_path.exists()


def test_plot_time_breakdown_barplot_hatches_only_the_unavailable_segment(
    two_result_entries_with_identical_generation_time_different_provenance: dict[
        str, CGComparisonResult
    ],
    legend_of: Callable[..., Legend],
) -> None:
    """A MEASURED and an UNAVAILABLE generation segment with identical wall time
    must render visually distinct -- only the UNAVAILABLE one is hatched.
    """
    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_time_breakdown_barplot(
            two_result_entries_with_identical_generation_time_different_provenance,
            legend_of("measured", "unavailable"),
        )
        fig = reporting_plots.plt.gcf()
        ax = fig.axes[0]
        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        # Each row draws 3 segments in order (generation, setup, solve); the
        # generation segment is the 1st patch (index 0) within each row's triple.
        generation_hatches_by_label = {
            label: ax.patches[i * 3].get_hatch() for i, label in enumerate(yticklabels)
        }
    reporting_plots.plt.close(fig)

    assert generation_hatches_by_label["Method measured"] is None
    assert generation_hatches_by_label["Method unavailable"] is not None


def test_plot_work_precision_uses_solve_time_even_when_generation_is_unavailable(
    two_result_entries_with_identical_generation_time_different_provenance: dict[
        str, CGComparisonResult
    ],
    legend_of: Callable[..., Legend],
) -> None:
    """Work precision plots measured solve time without an offline-cost gate."""
    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_work_precision(
            two_result_entries_with_identical_generation_time_different_provenance,
            legend=legend_of("measured", "unavailable"),
        )
        fig = reporting_plots.plt.gcf()
        ax = fig.axes[0]
        legend = ax.get_legend()
        plotted_labels = [text.get_text() for text in legend.get_texts()] if legend else []
        x_values = [
            float(offset[0])
            for collection in ax.collections
            for offset in cast(np.ndarray, collection.get_offsets())
        ]
        x_label = ax.get_xlabel()
    reporting_plots.plt.close(fig)

    assert plotted_labels == ["Method measured", "Method unavailable"]
    assert x_values == [0.1, 0.1]
    assert x_label == "Solve time (s)"


# ---------------------------------------------------------------------------
# plot_convergence_comparison — new parameters
# ---------------------------------------------------------------------------


def test_plot_convergence_comparison_with_title_and_params(
    tmp_path: Path,
    two_result_entries: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
) -> None:
    """plot_convergence_comparison with title/rtol/atol/max_iterations saves file.

    Args:
        tmp_path: Pytest temporary directory.
        two_result_entries: Two-entry result dict from fixture.
    """
    save_path = tmp_path / "conv.png"
    plot_convergence_comparison(
        two_result_entries,
        title=EditorialTitle("Test"),
        legend=legend_of(*two_result_entries),
        rtol=RTOL,
        atol=ATOL,
        max_iterations=MAX_ITERATIONS,
        save_path=save_path,
    )
    assert save_path.exists()


def test_plot_convergence_comparison_default_title_no_params(
    tmp_path: Path,
    two_result_entries: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
) -> None:
    """plot_convergence_comparison with defaults (no title/tolerances) saves file.

    Args:
        tmp_path: Pytest temporary directory.
        two_result_entries: Two-entry result dict from fixture.
    """
    save_path = tmp_path / "conv_default.png"
    plot_convergence_comparison(
        two_result_entries, legend=legend_of(*two_result_entries), save_path=save_path
    )
    assert save_path.exists()


def test_plot_convergence_comparison_draws_every_keyed_result_with_its_legend_entry(
    two_result_entries: dict[str, CGComparisonResult], legend_of: Callable[..., Legend]
) -> None:
    """Each independently keyed result gets its own curve and its own legend text."""
    legend = legend_of(*two_result_entries)

    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_convergence_comparison(two_result_entries, legend=legend)
        figure = reporting_plots.plt.gcf()

    axes = figure.axes[0]
    assert len(axes.lines) == len(two_result_entries)
    drawn = axes.get_legend()
    assert drawn is not None
    assert [text.get_text() for text in drawn.get_texts()] == [
        legend.label(name) for name in two_result_entries
    ]
    reporting_plots.plt.close(figure)


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


def test_plot_condition_numbers_saves_horizontal_chart(
    tmp_path: Path,
    simple_cond_numbers: dict[str, float],
) -> None:
    """plot_condition_numbers writes a PNG under save_dir and returns its path.

    Args:
        tmp_path: Pytest temporary directory.
        simple_cond_numbers: Simple condition number mapping from fixture.
    """
    result_path = plot_condition_numbers(simple_cond_numbers, save_dir=tmp_path)
    assert result_path is not None
    assert result_path.exists()


def test_plot_condition_numbers_with_title_and_subtitle(
    tmp_path: Path,
    simple_cond_numbers: dict[str, float],
) -> None:
    """plot_condition_numbers with title/rtol/atol writes a PNG and returns path.

    Args:
        tmp_path: Pytest temporary directory.
        simple_cond_numbers: Simple condition number mapping from fixture.
    """
    result_path = plot_condition_numbers(
        simple_cond_numbers,
        save_dir=tmp_path,
        title="My Comparison",
        rtol=RTOL,
        atol=ATOL,
    )
    assert result_path is not None
    assert result_path.exists()


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

    assert result.time_breakdown_barplot.exists()


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


def test_method_history_uses_metric_title_and_rendered_scope_subtitle(
    two_result_entries: dict[str, CGComparisonResult],
    legend_of: Callable[..., Legend],
    notation: NotationPolicy,
) -> None:
    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_convergence_comparison(
            two_result_entries,
            title=EditorialTitle("Residual History"),
            subtitle=ScopeSubtitle(AllDatasetsScope(), notation),
            legend=legend_of(*two_result_entries),
        )
        figure = reporting_plots.plt.gcf()

    axes = figure.axes[0]
    assert figure.get_suptitle() == "Residual History"
    assert axes.get_title() == "All datasets"
    reporting_plots.plt.close(figure)
