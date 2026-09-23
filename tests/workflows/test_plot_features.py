"""Tests for recently added plot features in plotting.py, spectra.py, and compare.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner

from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.domain.analysis.spectra import plot_condition_numbers
from neuralls.domain.solver.models.result import CGComparisonResult, PlotPaths
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.reporting import plots as reporting_plots
from neuralls.platform.reporting.plots import plot_convergence_comparison, plot_metric_comparison
from neuralls.platform.reporting.preconditioner_labels import build_preconditioner_labels

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
def simple_cond_numbers() -> dict[str, float]:
    """Simple condition number mapping for two preconditioners.

    Returns:
        Mapping from preconditioner name to condition number float.
    """
    return {"identity": 1.0, "jacobi": 5.0}


@pytest.fixture
def two_preconditioners() -> dict[str, Preconditioner]:
    """Constructed preconditioner instances matching ``two_result_entries``' keys.

    Returns:
        A dict with two entries: "identity" and "jacobi", holding real
        constructed preconditioner objects (not config data) so
        ``_generate_comparison_plots`` can derive descriptive plot labels
        from them.
    """
    matrix = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    return {
        "identity": Identity(),
        "jacobi": JacobiPreconditioner(matrix),
    }


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


# ---------------------------------------------------------------------------
# plot_convergence_comparison — new parameters
# ---------------------------------------------------------------------------


def test_plot_convergence_comparison_with_title_and_params(
    tmp_path: Path,
    two_result_entries: dict[str, CGComparisonResult],
) -> None:
    """plot_convergence_comparison with title/rtol/atol/max_iterations saves file.

    Args:
        tmp_path: Pytest temporary directory.
        two_result_entries: Two-entry result dict from fixture.
    """
    save_path = tmp_path / "conv.png"
    plot_convergence_comparison(
        two_result_entries,
        title="Test",
        rtol=RTOL,
        atol=ATOL,
        max_iterations=MAX_ITERATIONS,
        save_path=save_path,
    )
    assert save_path.exists()


def test_plot_convergence_comparison_default_title_no_params(
    tmp_path: Path,
    two_result_entries: dict[str, CGComparisonResult],
) -> None:
    """plot_convergence_comparison with defaults (no title/tolerances) saves file.

    Args:
        tmp_path: Pytest temporary directory.
        two_result_entries: Two-entry result dict from fixture.
    """
    save_path = tmp_path / "conv_default.png"
    plot_convergence_comparison(two_result_entries, save_path=save_path)
    assert save_path.exists()


def test_plot_convergence_comparison_draws_every_result_when_labels_collide(
    two_result_entries: dict[str, CGComparisonResult],
) -> None:
    """Equal display text must not collapse independently keyed result curves."""
    labels = dict.fromkeys(two_result_entries, "POD-2G (c=10)")

    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_convergence_comparison(two_result_entries, labels=labels)
        figure = reporting_plots.plt.gcf()

    axes = figure.axes[0]
    assert len(axes.lines) == len(two_result_entries)
    legend = axes.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == list(labels.values())
    reporting_plots.plt.close(figure)


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
    two_preconditioners: dict[str, Preconditioner],
    two_preconditioner_families: dict[str, PreconditionerType],
) -> None:
    """_generate_comparison_plots returns PlotPaths with iterations_barplot set.

    Plotting functions are patched to avoid file I/O while still exercising the
    orchestration logic that populates PlotPaths.

    Args:
        comparison_paths: Pytest-managed comparison output paths.
        two_result_entries: Two-entry result dict from fixture.
        two_preconditioners: Constructed preconditioner instances from fixture.
        two_preconditioner_families: Family key per name, from fixture.
    """
    labels = build_preconditioner_labels(two_preconditioners, two_preconditioner_families)

    with (
        patch(
            "neuralls.composition.comparison._plots.plot_convergence_comparison"
        ) as convergence_plot,
        patch(
            "neuralls.composition.comparison._plots.plot_error_convergence_comparison"
        ) as error_plot,
        patch("neuralls.composition.comparison._plots.plot_metric_comparison") as metric_plot,
    ):
        result = _generate_comparison_plots(
            two_result_entries,
            comparison_paths,
            labels,
            comparison_context="matrix=demo-matrix | rhs=gaussian",
            system_size=1000,
        )

    assert result.iterations_barplot is not None
    assert "condition_numbers" not in result.to_mapping()
    expected_title = "matrix=demo-matrix | rhs=gaussian\nN=1000"
    assert convergence_plot.call_args.kwargs["title"] == expected_title
    assert convergence_plot.call_args.args[0] is two_result_entries
    assert convergence_plot.call_args.kwargs["labels"] == labels
    assert error_plot.call_args.args[0] is two_result_entries
    assert error_plot.call_args.kwargs["labels"] == labels
    assert metric_plot.call_args.kwargs["title"] == expected_title


def test_generate_comparison_plots_preserves_results_when_labels_collide(
    comparison_paths: ComparisonPaths,
    two_result_entries: dict[str, CGComparisonResult],
    two_preconditioner_families: dict[str, PreconditionerType],
) -> None:
    """Presentation-label collisions must not overwrite distinct solver results."""
    labels = dict.fromkeys(two_result_entries, "POD-2G (c=10)")

    with (
        patch(
            "neuralls.composition.comparison._plots.plot_convergence_comparison"
        ) as convergence_plot,
        patch(
            "neuralls.composition.comparison._plots.plot_error_convergence_comparison"
        ) as error_plot,
        patch("neuralls.composition.comparison._plots.plot_metric_comparison"),
    ):
        _generate_comparison_plots(
            two_result_entries,
            comparison_paths,
            labels,
            comparison_context="matrix=demo-matrix | rhs=gaussian",
            families=two_preconditioner_families,
        )

    assert convergence_plot.call_args.args[0] is two_result_entries
    assert convergence_plot.call_args.kwargs["labels"] == labels
    assert error_plot.call_args.args[0] is two_result_entries
    assert error_plot.call_args.kwargs["labels"] == labels
