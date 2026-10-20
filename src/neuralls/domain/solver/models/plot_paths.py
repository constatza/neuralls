"""Paths of the diagnostic plots one comparison run generated."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PlotPaths:
    """Paths to generated diagnostic plots from a comparison workflow.

    Args:
        convergence: Convergence curve plot.
        parity: Parity plot (predicted vs true).
        residuals: Residual history plot.
        iterations_barplot: Horizontal bar chart of iteration counts.
        error_convergence: Relative energy-norm error convergence plot.
        generation_time_barplot: Horizontal bar chart of dataset-generation times.
        setup_time_barplot: Horizontal bar chart of preconditioner setup times.
        solve_time_barplot: Horizontal bar chart of CG solve times.
        avg_iteration_time_barplot: Horizontal bar chart of solve time per iteration.
        peak_memory_barplot: Horizontal bar chart of peak memory usage.
        time_breakdown_barplot: Stacked horizontal bar chart of generation vs.
            setup vs. solve time per method.
        work_precision: Work-precision diagram (final precision vs. solve time).
        residual_vs_time: Residual vs wall time (generation + setup + solve).
        error_vs_time: Energy-norm error vs wall time (when available).
        residual_vs_solve_time: Residual vs solve time only; every curve starts at t=0.
        error_vs_solve_time: Energy-norm error vs solve time only (when available).
    """

    convergence: Path | None = None
    parity: Path | None = None
    residuals: Path | None = None
    iterations_barplot: Path | None = None
    error_convergence: Path | None = None
    generation_time_barplot: Path | None = None
    setup_time_barplot: Path | None = None
    solve_time_barplot: Path | None = None
    avg_iteration_time_barplot: Path | None = None
    peak_memory_barplot: Path | None = None
    time_breakdown_barplot: Path | None = None
    work_precision: Path | None = None
    residual_vs_time: Path | None = None
    error_vs_time: Path | None = None
    residual_vs_solve_time: Path | None = None
    error_vs_solve_time: Path | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Path] | None) -> PlotPaths:
        """Build typed plot paths from a loose mapping, ignoring unknown keys.

        Args:
            mapping: Dict of plot type → path.

        Returns:
            PlotPaths with matched fields populated.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{key: path for key, path in (mapping or {}).items() if key in known})

    def to_mapping(self) -> dict[str, Path]:
        """Return only populated plot paths.

        Returns:
            Dict of plot type → path (omits None entries).
        """
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return {key: path for key, path in values.items() if path is not None}
