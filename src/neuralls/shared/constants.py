"""Constants and configuration values for graph-cg project.

This module centralizes magic values used across the codebase to improve
maintainability and avoid duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from neuralls.shared.types import MatrixNormType

# =============================================================================
# Exit Codes
# =============================================================================
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_KEYBOARD_INTERRUPT = 130

# =============================================================================
# Default Paths
# =============================================================================
# Dynamically determine project root relative to this file (src/neuralls/constants.py)
# src/neuralls/shared/constants.py -> src/neuralls/shared -> src/neuralls -> src -> project_root
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default Experiment Config Filenames
EXP_MODEL_CONFIG_NAME = "model.toml"
EXP_DATA_CONFIG_NAME = "data.toml"
EXP_SOLVER_CONFIG_NAME = "solver.toml"

# =============================================================================
# CG Solver Defaults
# =============================================================================
DEFAULT_RTOL = 1e-6  # Relative tolerance for CG solves
DEFAULT_ATOL = 1e-14  # Absolute tolerance for convergence (in addition to relative tolerance)

# Prediction diagnostic thresholds
PREDICTION_NORM_EPSILON = 1e-12  # Epsilon for safe division in prediction norm ratio

# FCG (Flexible Conjugate Gradient) Algorithm Parameters
DEFAULT_M_MAX = 20  # Maximum history length for truncated orthogonalization in FCG


# =============================================================================
# Data Generation Defaults
# =============================================================================
DEFAULT_NUM_SAMPLES = 6000

# Dataset artifacts (zarr dense format)
DATASET_MANIFEST_FILENAME = "manifest.json"
MATRIX_ZARR_DIRNAME = "matrix.zarr"
RHS_ZARR_FILENAME = "rhs.zarr"
SOLUTIONS_ZARR_FILENAME = "solutions.zarr"
PARAMETERS_ZARR_PREFIX = "parameters_"
"""Prefix for per-matrix parameters zarr dirs: ``parameters_0.zarr``, ``parameters_1.zarr``, …"""

# Strategy-specific iteration parameters
# These are now configured at the strategy level (not generation level):
# - DEFAULT_KRYLOV_ITERATIONS: Used by krylov strategy for Krylov subspace dimension
# - DEFAULT_RESIDUAL_TRACE_ITERS: Used by residual_traces, residuals, and gaussian_residuals strategies
DEFAULT_KRYLOV_ITERATIONS = 15
DEFAULT_RESIDUAL_TRACE_ITERS = 8

DEFAULT_RANDOM_SEED = 42
DEFAULT_SHUFFLE = True

# =============================================================================
# Comparison/Evaluation Defaults
# =============================================================================
DEFAULT_TEST_SAMPLE_INDEX = 0  # Which sample to extract for single-sample comparison tasks


# =============================================================================
# Eigenvector Selection
# =============================================================================
EIGENVECTOR_SELECT_SMALLEST = "smallest"
EIGENVECTOR_SELECT_LARGEST = "largest"
EIGENVECTOR_SELECT_RANDOM = "random"
EigenvectorSelectionMode = Literal["smallest", "largest", "random"]


# =============================================================================
# Matrix Norm Types (for dataset metadata)
# =============================================================================
# Convenience aliases for backward compatibility
MATRIX_NORM_SPECTRAL = MatrixNormType.SPECTRAL.value
MATRIX_NORM_FROBENIUS = MatrixNormType.FROBENIUS.value
MATRIX_NORM_NUCLEAR = MatrixNormType.NUCLEAR.value
MATRIX_NORM_ONE = MatrixNormType.ONE.value
MATRIX_NORM_INF = MatrixNormType.INF.value

# Default matrix norm for dataset metadata
DEFAULT_MATRIX_NORM_TYPE = MatrixNormType.SPECTRAL


# =============================================================================
# File I/O
# =============================================================================
FILE_MODE_READ_BINARY = "rb"
FILE_MODE_WRITE_BINARY = "wb"
FILE_MODE_READ_TEXT = "r"
FILE_MODE_WRITE_TEXT = "w"
FILE_ENCODING_UTF8 = "utf-8"

# =============================================================================
# UI/Output Symbols
# =============================================================================
SYMBOL_SUCCESS = "✓"
SYMBOL_ERROR = "✗"
SYMBOL_ROCKET = "🚀"
SYMBOL_CHART = "📊"
SYMBOL_CHECKMARK = "✅"
SYMBOL_WARNING = "⚠️"

# =============================================================================
# Noise Analysis
# =============================================================================
NOISE_STRATEGY_NONE = "none"
DEFAULT_NOISE_LEVEL = 0.05
DEFAULT_NOISE_SEED = None

# =============================================================================
# Validation Thresholds
# =============================================================================
MIN_MATRIX_SIZE = 1
MIN_TOLERANCE = 1e-15
MAX_ITERATIONS_UPPER_LIMIT = 1_000_000

# =============================================================================
# Plot Settings
# =============================================================================
DEFAULT_PLOT_DPI = 150
DEFAULT_PLOT_FIGSIZE = (10, 6)
DEFAULT_PLOT_STYLE = "seaborn-v0_8-darkgrid"
