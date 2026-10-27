"""Tests for comparison log and plot presentation context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison._presentation import (
    build_comparison_source_context,
)
from neuralls.composition.comparison.models import ComparisonPaths, ResolvedComparisonInput
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.shared.types import ComparisonRhsSourceKind

WORKSPACE = Path("/workspace")


@pytest.fixture
def comparison_paths() -> ComparisonPaths:
    """Return resolved paths for one comparison run."""
    return ComparisonPaths(
        matrix=WORKSPACE / "gaussian-cg10-spheres-1000x",
        rhs=WORKSPACE / "resolved-rhs.npy",
        output=WORKSPACE / "output",
        figures=WORKSPACE / "output" / "figures",
    )


@pytest.fixture
def gaussian_comparison_input() -> ResolvedComparisonInput:
    """Return a generated-Gaussian comparison input with explicit provenance."""
    return ResolvedComparisonInput(
        matrix=np.eye(2),
        rhs=np.ones(2),
        matrix_dataset_id="gaussian-cg10-spheres-1000x",
        matrix_index=0,
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
    )


def test_context_names_matrix_and_rhs_for_logs(
    comparison_paths: ComparisonPaths,
    gaussian_comparison_input: ResolvedComparisonInput,
) -> None:
    """The log context states the matrix dataset and the RHS source."""
    context = build_comparison_source_context(
        comparison_paths,
        gaussian_comparison_input,
    )

    assert context == "matrix=gaussian-cg10-spheres-1000x | rhs=gaussian"
