"""Workflow configuration and reporting models for solver orchestration."""

from .cg_result import CGComparisonResult
from .config import ComparisonData, ComparisonGeneral, SolverParams
from .plot_paths import PlotPaths
from .recommendations import ComparisonRecommendations, RankedRecommendation
from .result import ComparisonResult

__all__ = [
    "CGComparisonResult",
    "ComparisonData",
    "ComparisonGeneral",
    "ComparisonRecommendations",
    "ComparisonResult",
    "PlotPaths",
    "RankedRecommendation",
    "SolverParams",
]
