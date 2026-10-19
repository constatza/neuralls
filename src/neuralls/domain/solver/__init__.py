"""Solver workflow models and comparison orchestration.

Solver and preconditioner algorithms are owned by :mod:`torchalg`. This package
keeps neuralls workflow DTOs, configuration models, validation helpers, and
comparison/reporting orchestration.
"""

from __future__ import annotations

from torchalg.monitoring import TraceMode

from neuralls.domain.solver.comparison import run_cg_comparison
from neuralls.domain.solver.comparison_summary import (
    format_results_summary,
    summarize_best_combinations,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.domain.solver.models.recommendations import (
    ComparisonRecommendations,
    RankedRecommendation,
)
from neuralls.domain.solver.models.result import ComparisonResult

from .models.config import ComparisonData, ComparisonGeneral, SolverConfig, SolverParams

__all__ = [
    "CGComparisonResult",
    "ComparisonData",
    "ComparisonGeneral",
    "ComparisonRecommendations",
    "ComparisonResult",
    "PlotPaths",
    "RankedRecommendation",
    "SolverConfig",
    "SolverParams",
    "TraceMode",
    "format_results_summary",
    "run_cg_comparison",
    "summarize_best_combinations",
]

__version__ = "2.0.0"
__author__ = "neuralls Contributors"
__description__ = "Modular Flexible CG solver with strategy pattern design"
