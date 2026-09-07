"""Data-config driven helpers for collection and generation scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import ValidationError

from neuralls.composition.generation.processing import process_config
from neuralls.domain.generation.source_streams import _is_glob_expression
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.base import load_matrix


def process_data_from_config(
    config_path: Path,
    settings: NeurallsSettings,
    *,
    force: bool = False,
) -> Path:
    """Unified entry point for data collection and generation.

    This function handles both data collection (from existing RHS archives)
    and synthetic data generation through the same unified pipeline.

    Args:
        config_path: Path to TOML data configuration file.
        settings: Resolved runtime settings.
        force: Regenerate even if a matching dataset already exists.

    Returns:
        Path to output dataset directory.
    """
    config = load_data_config(config_path, settings)
    if config.source.matrix_path is None:
        raise ValueError("Missing 'source.matrix_path' in config")

    if config.output.data_dir is None:
        config = config.model_copy(
            update={"output": config.output.with_data_dir(settings.processed_dir)}
        )

    matrix_path = config.source.matrix_path
    if matrix_path is None:
        raise ValueError("Missing 'source.matrix_path' in config")

    # Glob paths fan out to multiple matrix files; GlobMatrixStream handles loading.
    # Eager single-file load is only needed for the provide_rhs=True edge case.
    matrix: np.ndarray | None = (
        None if _is_glob_expression(matrix_path) else load_matrix(Path(matrix_path))
    )
    try:
        return process_config(config, matrix, force=force)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
