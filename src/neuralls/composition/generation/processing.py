"""Unified data processing entry point for the generate workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuralls.composition.generation._context_builder import _build_context
from neuralls.composition.generation._strategy_executor import _execute_plan
from neuralls.platform.config.models.data_models import DataConfigFile


def process_config(
    config: DataConfigFile,
    matrix: np.ndarray | None = None,
    *,
    force: bool = False,
) -> Path:
    """Process a data config and execute the declared generation plan.

    Args:
        config: Fully resolved DataConfigFile (output.data_dir must be set).
        matrix: Optional pre-loaded system matrix. When None the matrix is
            loaded from config.source.matrix_path during generation.
        force: Regenerate even if a matching dataset already exists.

    Returns:
        Path to the generated dataset directory.
    """
    context, plan = _build_context(config=config)
    dataset_dir = _execute_plan(context, config.generation, plan, matrix, force=force)

    if config.test.solutions_path:
        print("\n=== Skipping comparison split generation (not yet migrated) ===")

    return dataset_dir
