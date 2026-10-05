"""Solution Archive Strategy using SOLID provider + transform pattern.

Architecture:
    Layer 1 (Input): FileInputProvider loads solution vectors from disk
    Layer 2 (Transform): ComputeRhsTransform computes b = A @ x
    Layer 3 (Strategy): Orchestrates provider and transform
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import FileInputProvider
from ..runner import register_strategy
from ..strategy_configs import SolutionArchiveConfig
from ..transforms import ComputeRhsTransform


@register_strategy
class SolutionArchiveStrategy:
    """Load solution vectors from archive and compute RHS = A @ x.

    SOLID Pattern:
        - FileInputProvider: Loads solutions from disk (Layer 1)
        - ComputeRhsTransform: Computes b = A @ x (Layer 2)
        - Strategy: Orchestrates provider and transform (Layer 3)

    This decoupling enables:
        - Replacing file input with in-memory archive
        - Reusing ComputeRhsTransform across strategies
        - Testing provider and transform independently

    Configuration:
        - solutions_glob (str): Glob pattern for solution files (e.g., "/data/sol_*.txt")
        - samples (int): Number of files to load (-1 for all)
        - shuffle (bool): Whether to shuffle file selection (default: False)
        - seed (int | None): Random seed for shuffling (required if shuffle=True)

    Examples:
        >>> # Load first 50 solution files
        >>> cfg = {
        ...     "solutions_glob": "/data/solutions_*.txt",
        ...     "samples": 50,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)

        >>> # Load all solution files with shuffling
        >>> cfg = {
        ...     "solutions_glob": "/data/sol_*.npy",
        ...     "samples": -1,
        ...     "shuffle": True,
        ...     "seed": 42,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)
    """

    name = "solution_archive"
    ConfigType = SolutionArchiveConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Load solution vectors from archive and compute RHS = A @ x.

        Args:
            matrix: System matrix (already normalized by orchestrator)
            cfg: Configuration dictionary with keys:
                - solutions_glob: Glob pattern for solution files
                - samples: Number of files to load
                - shuffle: Whether to shuffle file selection
                - seed: Random seed for shuffling
            archive: Optional pre-loaded archive data (ignored by this strategy as it loads from disk)

        Returns:
            GeneratedSamples with loaded solutions and computed RHS

        Raises:
            ValueError: If solutions_glob not provided or insufficient files
            FileNotFoundError: If no files match pattern
        """
        # When archive is provided (per-binding pre-loaded solution), skip glob loading
        if archive is not None and archive.lhs is not None:
            solutions = archive.lhs  # shape (1, n) for single-binding case
            logger.info(f"Using pre-loaded solution archive ({len(solutions)} vector(s))")
            rhs = ComputeRhsTransform(matrix).transform(solutions)
            return GeneratedSamples(
                matrix=matrix,
                rhs=rhs,
                solutions=solutions,
                residual_traces=None,
                error_traces=None,
            )

        # Validate and convert to typed config
        config = SolutionArchiveConfig(**cfg)

        # Extract configuration
        solutions_glob = config.solutions_glob
        samples = config.samples
        shuffle = config.shuffle
        seed = config.seed
        skip = config.skip

        logger.info(f"Loading solution vectors from archive: {solutions_glob}")

        # Layer 1: Input provision (load from files)
        provider = FileInputProvider(
            glob_pattern=solutions_glob,
            shuffle=shuffle,
            seed=seed,
            skip=skip,
            file_indices=config.file_indices,
        )
        rng = np.random.default_rng(seed)
        solutions = provider.provide(matrix, count=samples, rng=rng)

        logger.info(f"Loaded {len(solutions)} solution vectors")

        # Layer 2: Transformation (compute RHS from solutions)
        logger.info(f"Computing RHS for {len(solutions)} solutions...")
        transform = ComputeRhsTransform(matrix)
        rhs = transform.transform(solutions)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs,
            solutions=solutions,
            residual_traces=None,
            error_traces=None,
        )
