"""Scaled Solutions Strategy using SOLID provider + transform pattern.

Architecture:
    Layer 1 (Input): FileInputProvider loads solution vectors from disk
    Layer 2 (Transform): ComputeRhsTransform computes b = A @ x
    Layer 3 (Strategy): Applies scale factor and returns (scale*b, scale*x)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import FileInputProvider
from ..runner import register_strategy
from ..strategy_configs import ScaledSolutionsConfig
from ..transforms import ComputeRhsTransform


@register_strategy
class ScaledSolutionsStrategy:
    """Load solution vectors from archive and return scaled (b, x) pairs.

    SOLID Pattern:
        - FileInputProvider: Loads solutions from disk (Layer 1)
        - ComputeRhsTransform: Computes b = A @ x (Layer 2)
        - Strategy: Applies scale factor to both sides (Layer 3)

    The scaling identity A @ (scale * x) = scale * b is preserved,
    so all returned pairs are valid training samples.

    Configuration:
        - solutions_glob (str): Glob pattern for solution files
        - scale (float): Scalar multiplier applied to all (x, b) pairs (default: 5.0)
        - samples (int): Number of files to load (-1 for all)
        - shuffle (bool): Whether to shuffle file selection (default: False)
        - seed (int | None): Random seed for shuffling

    Examples:
        >>> cfg = {
        ...     "solutions_glob": "/data/solutions_*.txt",
        ...     "scale": 5.0,
        ...     "samples": 50,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)
    """

    name = "scaled_solutions"
    ConfigType = ScaledSolutionsConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Load solutions, compute RHS, and return scale * (b, x) pairs.

        Args:
            matrix: System matrix (already normalized by orchestrator).
            cfg: Configuration dictionary with keys:
                - solutions_glob: Glob pattern for solution files
                - scale: Scalar multiplier applied to all (x, b) pairs
                - samples: Number of files to load
                - shuffle: Whether to shuffle file selection
                - seed: Random seed for shuffling
            archive: Optional pre-loaded archive data (ignored; loads from disk).

        Returns:
            GeneratedSamples with scaled solutions and scaled RHS.

        Raises:
            ValueError: If solutions_glob not provided or insufficient files.
            FileNotFoundError: If no files match pattern.
        """
        # When archive is provided (per-binding pre-loaded solution), skip glob loading
        if archive is not None and archive.lhs is not None:
            solutions = archive.lhs
            logger.info(f"Using pre-loaded solution archive ({len(solutions)} vector(s))")
            rhs = ComputeRhsTransform(matrix).transform(solutions)
            scale = cfg.get("scale", 5.0)
            scaled_rhs = float(scale) * rhs
            scaled_solutions = float(scale) * solutions
            return GeneratedSamples(
                matrix=matrix,
                rhs=scaled_rhs,
                solutions=scaled_solutions,
                residual_traces=None,
                error_traces=None,
            )

        config = ScaledSolutionsConfig(**cfg)

        logger.info(f"Loading solution vectors from: {config.solutions_glob}")

        # Layer 1: load solutions from disk
        provider = FileInputProvider(
            glob_pattern=config.solutions_glob,
            shuffle=config.shuffle,
            seed=config.seed,
            skip=config.skip,
            file_indices=config.file_indices,
        )
        rng = np.random.default_rng(config.seed)
        solutions = provider.provide(matrix, count=config.samples, rng=rng)

        logger.info(f"Loaded {len(solutions)} solution vectors")

        # Layer 2: compute RHS = A @ x
        logger.info(f"Computing RHS for {len(solutions)} solutions...")
        rhs = ComputeRhsTransform(matrix).transform(solutions)

        # Layer 3: apply scale factor (A @ (scale*x) = scale*b)
        logger.info(f"Applying scale factor {config.scale}...")
        scaled_rhs = config.scale * rhs
        scaled_solutions = config.scale * solutions

        return GeneratedSamples(
            matrix=matrix,
            rhs=scaled_rhs,
            solutions=scaled_solutions,
            residual_traces=None,
            error_traces=None,
        )
