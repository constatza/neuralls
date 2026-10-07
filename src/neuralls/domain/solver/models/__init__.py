"""Workflow configuration and reporting models for solver orchestration."""

from .config import ComparisonData, ComparisonGeneral, SolverParams
from .result import (
    CGComparisonResult,
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
    RankedRecommendation,
)

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
