"""Shared domain types used across multiple neuralls subpackages.

Types placed here satisfy two criteria:
- They carry no domain logic (pure data / TypedDict / StrEnum)
- They are imported by more than one top-level subpackage

Moving cross-boundary types here prevents upward dependencies, e.g. io → generation.
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Literal, TypedDict

DatasetFormat = Literal["zarr", "npy", "hdf5"]
"""Supported dataset storage families for generated datasets."""

EntryRole = Literal["feature", "target"]
"""Runtime dataset entry roles understood by neuralls bridge code."""


class LayoutType(StrEnum):
    """Physical matrix layout in a generated dataset."""

    MANY_MATRICES = "many_matrices"
    BROADCAST_SINGLE = "broadcast_single"


class MatrixNormType(StrEnum):
    """Matrix norm types for dataset metadata."""

    SPECTRAL = "spectral"
    FROBENIUS = "frobenius"
    NUCLEAR = "nuclear"
    ONE = "one"
    INF = "inf"


class RowKind(int, Enum):
    """Semantic category of a dataset row."""

    STANDARD = 0
    CG_INTERNAL = 1

    def __str__(self) -> str:
        """Return the lowercase name of the enum member."""
        return self.name.lower()


class ComparisonRhsSourceKind(StrEnum):
    """Supported RHS sources for comparison workflows."""

    GAUSSIAN = "gaussian"
    SPARSE = "sparse"
    RAW_LHS = "raw_lhs"
    RAW_RHS = "raw_rhs"
    DATASET = "dataset"


class PreconditionerFamily(StrEnum):
    """Plot-style grouping for preconditioner variants that share a config `type`.

    ``PreconditionerType.AMG`` covers both classical AMG and POD-2G coarsening,
    and ``NEURAL``/``NEURAL_AMG`` are both "neural" for comparison-plot
    purposes — these three names give those merged groups a real, checkable
    identity instead of ad hoc strings. Every other preconditioner type is
    already its own distinct group, so it uses its own `PreconditionerType`
    member directly rather than duplicating it here (see
    ``platform.config.models.preconditioner_family.preconditioner_family``).
    """

    AMG = "amg"
    POD2G = "pod2g"
    NEURAL = "neural"


class GenerationStrategyKind(StrEnum):
    """Canonical generation strategy identifiers used after config validation."""

    RANDOM = "random"
    NORMAL = "normal"
    KRYLOV = "krylov"
    RHS_ARCHIVE = "rhs_archive"
    SOLUTION_ARCHIVE = "solution_archive"
    VALIDATED_ARCHIVE = "validated_archive"
    SCALED_SOLUTIONS = "scaled_solutions"
    SPARSE_RHS = "sparse_rhs"
    RESIDUALS = "residuals"
    GAUSSIAN_RESIDUALS = "gaussian_residuals"
    SEARCH_DIRECTIONS = "search_directions"
    EIGENVECTOR_FORWARD = "eigenvector_forward"
    EIGENVECTOR_INVERSE = "eigenvector_inverse"
    GAUSSIAN_FORWARD = "gaussian_forward"
    GAUSSIAN_INVERSE = "gaussian_inverse"
    UNIFORM_FORWARD = "uniform_forward"
    UNIFORM_INVERSE = "uniform_inverse"
    CONSTANT_FORWARD = "constant_forward"
    CONSTANT_INVERSE = "constant_inverse"
    NEUTRAL_ONES = "neutral_ones"
    SMOOTHER_FILTERED_PROBES = "smoother_filtered_probes"


class ScaleMetadata(TypedDict, total=False):
    """Type-safe schema for scale metadata dictionary."""

    spectral_radius_bound: float
    dimension_scale: float
