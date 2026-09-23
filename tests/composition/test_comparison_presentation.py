"""Tests for comparison log and plot presentation context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison._presentation import (
    build_comparison_plot_title,
    build_comparison_source_context,
)
from neuralls.composition.comparison.models import ComparisonPaths, ResolvedComparisonInput
from neuralls.shared.types import ComparisonRhsSourceKind


@pytest.fixture
def comparison_paths(tmp_path: Path) -> ComparisonPaths:
    """Return resolved paths for one comparison run."""
    return ComparisonPaths(
        matrix=tmp_path / "gaussian-cg10-spheres-1000x",
        rhs=tmp_path / "resolved-rhs.npy",
        output=tmp_path / "output",
        figures=tmp_path / "output" / "figures",
    )


@pytest.fixture
def gaussian_comparison_input() -> ResolvedComparisonInput:
    """Return a generated-Gaussian comparison input with explicit provenance."""
    return ResolvedComparisonInput(
        matrix=np.eye(2),
        rhs=np.ones(2),
        matrix_dataset_id="gaussian-cg10-spheres-1000x",
        matrix_index=0,
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
    )


def test_context_is_shared_by_logs_and_plot_titles(
    comparison_paths: ComparisonPaths,
    gaussian_comparison_input: ResolvedComparisonInput,
) -> None:
    """The canonical context is explicit and the plot title adds only system size."""
    context = build_comparison_source_context(
        comparison_paths,
        gaussian_comparison_input,
    )

    assert context == "matrix=gaussian-cg10-spheres-1000x | rhs=gaussian"
    assert (
        build_comparison_plot_title(context, 1000)
        == "matrix=gaussian-cg10-spheres-1000x | rhs=gaussian\nN=1000"
    )
