"""Data generation module for synthetic linear system datasets.

This module provides a complete framework for generating synthetic training data
for linear system solvers. It supports multiple generation strategies (normal,
krylov, residual traces, error traces, eigenvector-based) that can be mixed
according to user-specified proportions.

Public API:
    generate_mixture: Main orchestration function for mixed-strategy generation
    run_generation: Registry-based strategy dispatcher
    Data types: StrategyOutput, ArchiveData, GeneratedSamples
    Helpers: rng_from_seed, rounded_counts

Architecture:
    - types: Immutable data structures
    - helpers: Pure utility functions
    - trace_utils: Trace manipulation
    - orchestration: High-level workflow (generate_mixture)
    - strategies: Individual generation strategies (registry pattern)
    - runner: Strategy registry and dispatcher

Usage:
    >>> from neuralls.domain.generation import generate_mixture
    >>> X, Y, res_traces, err_traces = generate_mixture(
    ...     A, b, mix={"normal": 1.0, "krylov": 1.0}, total=100, seed=42
    ... )
"""

from __future__ import annotations

# SOLID Architecture (Phases 1-2 complete)
# Import strategies to trigger registration
from . import providers, strategies, transforms
from .data_types import NormalizeType
from .helpers import rng_from_seed, rounded_counts, select_archive_files
from .orchestration import build_dataset_payload, generate_mixture
from .payloads import GeneratedDatasetPayload
from .plan import GenerationPlan, StrategySpec, parse_generation_plan
from .runner import run_generation
from .specs import DatasetSpec, MixtureSpec, SourceSpec
from .types import ArchiveData, GeneratedSamples, StrategyOutput

__all__ = [
    "ArchiveData",
    "DatasetSpec",
    "GeneratedDatasetPayload",
    "GeneratedSamples",
    "GenerationPlan",
    "MixtureSpec",
    "NormalizeType",
    "SourceSpec",
    # Data types
    "StrategyOutput",
    "StrategySpec",
    "build_dataset_payload",
    # Main API
    "generate_mixture",
    "parse_generation_plan",
    # SOLID Components (Phase 1-2)
    "providers",
    # Helpers
    "rng_from_seed",
    "rounded_counts",
    "run_generation",
    "select_archive_files",
    # Strategies (for registration)
    "strategies",
    "transforms",
]

__version__ = "2.0.0"
__author__ = "neuralls Contributors"
__description__ = "SOLID-compliant data generation framework"
