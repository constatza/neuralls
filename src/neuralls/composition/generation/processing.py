"""Unified data processing entry point for the generate workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neuralls.composition.generation._context_builder import _build_context
from neuralls.composition.generation._strategy_executor import _execute_plan
from neuralls.composition.identity.generation import generation_identity
from neuralls.domain.identity import StageIdentity
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.shared.types import MatrixFormat, SparsityPattern


def process_config(
    config: DataConfigFile,
    matrix: np.ndarray | None = None,
    *,
    force: bool = False,
    identity: StageIdentity | None = None,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
    sparsity_pattern: SparsityPattern = SparsityPattern.RAGGED,
) -> Path:
    """Process a data config and execute the declared generation plan.

    Args:
        config: Fully resolved DataConfigFile (output.data_dir must be set).
        matrix: Optional pre-loaded system matrix. When None the matrix is
            loaded from config.source.matrix_path during generation.
        force: Regenerate even if a matching dataset already exists.
        identity: Precomputed generation identity; derived from `config` when None.
        matrix_format: Storage format of the system matrices (`[output].matrix_format`).
        sparsity_pattern: Layout of the stored CSR samples (`[output].sparsity_pattern`).

    Returns:
        Path to the generated dataset directory.
    """
    context, plan = _build_context(config=config)
    context = replace(context, identity=identity or generation_identity(config))
    dataset_dir = _execute_plan(
        context,
        config.generation,
        plan,
        matrix,
        force=force,
        matrix_format=matrix_format,
        sparsity_pattern=sparsity_pattern,
    )

    if config.test.solutions_path:
        print("\n=== Skipping comparison split generation (not yet migrated) ===")

    return dataset_dir
