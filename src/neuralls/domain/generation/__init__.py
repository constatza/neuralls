"""Data generation module for synthetic linear system datasets.

This module provides a complete framework for generating synthetic training data
for linear system solvers. It supports multiple generation strategies (normal,
krylov, residual traces, error traces, eigenvector-based) that can be mixed
according to user-specified proportions.

Public API:
    open_batch_stream: Streamed mixed-strategy generation (plan, then batches)
    run_generation: Registry-based strategy dispatcher
    Data types: StrategyOutput, ArchiveData, GeneratedSamples
    Helpers: rng_from_seed, rounded_counts

Architecture:
    - types: Immutable data structures
    - helpers: Pure utility functions
    - trace_utils: Trace manipulation
    - orchestration: High-level streamed workflow (open_batch_stream)
    - strategies: Individual generation strategies (registry pattern)
    - runner: Strategy registry and dispatcher

Usage:
    Build datasets through composition.generation.dataset_builder.build_dataset, which
    plans with open_batch_stream and writes batches as they are produced.
"""

from __future__ import annotations

# SOLID Architecture (Phases 1-2 complete)
# Import strategies to trigger registration
from . import providers, strategies, transforms
from .archive_files import select_archive_files
from .counts import rounded_counts
from .data_types import NormalizeType
from .orchestration import open_batch_stream
from .plan import GenerationPlan, StrategySpec, parse_generation_plan
from .runner import run_generation
from .seeds import rng_from_seed
from .specs import DatasetSpec, MixtureSpec, SourceSpec
from .types import ArchiveData, GeneratedSamples, StrategyOutput

__all__ = [
    "ArchiveData",
    "DatasetSpec",
    "GeneratedSamples",
    "GenerationPlan",
    "MixtureSpec",
    "NormalizeType",
    "SourceSpec",
    # Data types
    "StrategyOutput",
    "StrategySpec",
    # Main API
    "open_batch_stream",
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
