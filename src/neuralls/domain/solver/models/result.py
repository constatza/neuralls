"""Typed result of one preconditioner comparison workflow run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.recommendations import ComparisonRecommendations

if TYPE_CHECKING:
    from neuralls.domain.solver.models.config import ComparisonGeneral
    from neuralls.shared.types import ComparisonRhsSourceKind


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Typed result from compare_preconditioners workflow.

    Args:
        results: Raw CG comparison results keyed by preconditioner name.
        summary: Formatted text summary of results.
        solver_params: Solver parameters and data context used.
        plot_paths: Paths to generated diagnostic plots.
        preconditioners: Preconditioner names tested.
        recommendations: Ranked recommendations.
        output_dir: Root output directory for comparison artifacts.
        matrix_shape: Shape of the system matrix A, e.g. ``(n, n)``.
        rhs_shape: Shape of the right-hand side vector b, e.g. ``(n,)``.
        rhs_source_kind: Provenance of the RHS (e.g. gaussian, sparse, raw_lhs,
            raw_rhs, dataset) — a matrix may be paired with several different
            RHS sources across sibling ``[[comparisons]]`` entries, each its
            own MLflow run, so this disambiguates which one a given run used.
        plot_styles: How each preconditioner is drawn, by name, so later
            group-level plots reuse the per-run styling.
    """

    results: Mapping[str, CGComparisonResult]
    summary: str
    solver_params: ComparisonGeneral
    plot_paths: PlotPaths = field(default_factory=PlotPaths)
    preconditioners: tuple[str, ...] = ()
    recommendations: ComparisonRecommendations = field(default_factory=ComparisonRecommendations)
    output_dir: Path | None = None
    matrix_shape: tuple[int, int] | None = None
    rhs_shape: tuple[int, ...] | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
    plot_styles: Mapping[str, PreconditionerPlotStyle] = field(default_factory=dict)
