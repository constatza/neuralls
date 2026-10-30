"""Workflow configuration and reporting models for solver orchestration."""

from .benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    PeakMemory,
    SetupCompleted,
    SetupFailed,
    SetupOutcome,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SolveOutcome,
    SystemTag,
    UnknownPeakMemory,
)
from .cg_result import CGComparisonResult
from .config import ComparisonData, ComparisonGeneral, SolverParams
from .plot_paths import PlotPaths
from .recommendations import ComparisonRecommendations, RankedRecommendation

__all__ = [
    "CGComparisonResult",
    "ComparisonData",
    "ComparisonGeneral",
    "ComparisonRecommendations",
    "MatrixSetupEvent",
    "MeasuredPeakMemory",
    "PeakMemory",
    "PlotPaths",
    "RankedRecommendation",
    "SetupCompleted",
    "SetupFailed",
    "SetupOutcome",
    "SolveAttempt",
    "SolveConverged",
    "SolveFailed",
    "SolveOutcome",
    "SolverParams",
    "SystemTag",
    "UnknownPeakMemory",
]
