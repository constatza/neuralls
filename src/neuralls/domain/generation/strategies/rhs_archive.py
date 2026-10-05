"""RHS Archive Strategy using SOLID provider + transform pattern.

Architecture:
    Layer 1 (Input): FileInputProvider loads RHS vectors from disk
    Layer 2 (Transform): SolveTransform computes x = A^-1 @ b
    Layer 3 (Strategy): Orchestrates provider and transform
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import FileInputProvider
from ..runner import register_strategy
from ..strategy_configs import RhsArchiveConfig
from ..transforms import SolveTransform


@register_strategy
class RhsArchiveStrategy:
    """Load RHS vectors from archive and solve systems.

    SOLID Pattern:
        - FileInputProvider: Loads RHS from disk (Layer 1)
        - SolveTransform: Computes x = A^-1 @ b (Layer 2)
        - Strategy: Orchestrates provider and transform (Layer 3)

    This decoupling enables:
        - Replacing file input with in-memory archive
        - Reusing SolveTransform across strategies
        - Testing provider and transform independently
        - Switching between CG and direct solvers

    Configuration:
        - rhs_glob (str): Glob pattern for RHS files (e.g., "/data/rhs_*.txt")
        - samples (int): Number of files to load (-1 for all)
        - shuffle (bool): Whether to shuffle file selection (default: False)
        - seed (int | None): Random seed for shuffling (required if shuffle=True)
        - solve_config (SolveConfig): Configuration for solving systems (rtol, atol, etc.)

    Note:
        This strategy ALWAYS solves systems to produce valid (b, A^-1@b) training pairs.

    Examples:
        >>> # Load first 100 RHS files and solve with specific tolerances
        >>> cfg = {
        ...     "rhs_glob": "/data/rhs_*.txt",
        ...     "samples": 100,
        ...     "solve_config": {"method": "cg", "rtol": 1e-12, "atol": 1e-15},
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)

        >>> # Load all RHS files with shuffling
        >>> cfg = {
        ...     "rhs_glob": "/data/rhs_*.txt",
        ...     "samples": -1,
        ...     "shuffle": True,
        ...     "seed": 42,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)
    """

    name = "rhs_archive"
    ConfigType = RhsArchiveConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Load RHS vectors from archive and solve systems.

        Args:
            matrix: System matrix (already normalized by orchestrator)
            cfg: Configuration dictionary with keys:
                - rhs_glob: Glob pattern for RHS files
                - samples: Number of files to load
                - shuffle: Whether to shuffle file selection
                - seed: Random seed for shuffling
                - solve_config: SolveConfig or dict with solve parameters
            archive: Optional pre-loaded archive data (ignored by this strategy as it loads from disk)

        Returns:
            GeneratedSamples with loaded RHS and solved solutions

        Raises:
            ValueError: If rhs_glob not provided or insufficient files
            FileNotFoundError: If no files match pattern
        """
        # Validate and convert to typed config
        config = RhsArchiveConfig(**cfg)

        # Extract configuration
        rhs_glob = config.rhs_glob
        samples = config.samples
        shuffle = config.shuffle
        seed = config.seed
        solve_cfg = config.solve_config

        logger.info(f"Loading RHS vectors from archive: {rhs_glob}")

        # Layer 1: Input provision (load from files)
        provider = FileInputProvider(
            glob_pattern=rhs_glob,
            shuffle=shuffle,
            seed=seed,
            file_indices=config.file_indices,
        )
        rng = np.random.default_rng(seed)
        rhs = provider.provide(matrix, count=samples, rng=rng)

        logger.info(f"Loaded {len(rhs)} RHS vectors")

        # Layer 2: Transformation (solve systems)
        logger.info(f"Solving {len(rhs)} linear systems using {solve_cfg.method}...")
        transform = SolveTransform(
            matrix,
            method=solve_cfg.method,
            rtol=solve_cfg.rtol,
            atol=solve_cfg.atol,
            max_iters=solve_cfg.max_iters,
            assume_pos_def=solve_cfg.assume_pos_def,
        )
        solutions = transform.transform(rhs)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs,
            solutions=solutions,
            residual_traces=None,
            error_traces=None,
        )
