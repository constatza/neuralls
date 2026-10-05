"""Validated Archive Strategy using SOLID provider + transform pattern.

Architecture:
    Layer 1 (Input): PairedFileInputProvider loads (x, b) pairs from disk
    Layer 2 (Transform): VerifyTransform validates ||A @ x - b|| < tolerance
    Layer 3 (Strategy): Orchestrates provider and transform
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import PairedFileInputProvider
from ..runner import register_strategy
from ..strategy_configs import ValidatedArchiveConfig
from ..transforms import VerifyTransform


@register_strategy
class ValidatedArchiveStrategy:
    """Load pre-computed (x, b) pairs from archive and verify consistency.

    SOLID Pattern:
        - PairedFileInputProvider: Loads (x, b) pairs from disk (Layer 1)
        - VerifyTransform: Validates ||A @ x - b|| < tolerance (Layer 2)
        - Strategy: Orchestrates provider and transform (Layer 3)

    This decoupling enables:
        - Replacing file input with in-memory archive
        - Reusing VerifyTransform across strategies
        - Testing provider and transform independently
        - Adding pre-validation filters

    This strategy loads both solution vectors and RHS vectors from disk,
    then verifies that A @ x = b holds for all pairs (up to tolerance).
    This avoids wasted computation when archives contain both components.

    Use cases:
        - Validating collected data before training
        - Loading pre-computed pairs without redundant computation
        - Quality checking archived datasets

    Configuration:
        - solutions_glob: Glob pattern for solution files
        - rhs_glob: Glob pattern for RHS files
        - samples: Number of pairs to load (-1 for all)
        - shuffle: Whether to shuffle file selection
        - seed: Random seed for shuffling
        - verification_tolerance: Maximum relative residual (default: 1e-10)
        - fail_on_invalid: Whether to raise error on invalid pairs (default: True)

    Examples:
        >>> # Load and verify first 100 pairs
        >>> cfg = {
        ...     "solutions_glob": "/data/solutions_*.txt",
        ...     "rhs_glob": "/data/rhs_*.txt",
        ...     "samples": 100,
        ...     "verification_tolerance": 1e-10,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)

        >>> # Load all pairs with shuffling, warn on invalid
        >>> cfg = {
        ...     "solutions_glob": "/data/sol_*.npy",
        ...     "rhs_glob": "/data/rhs_*.npy",
        ...     "samples": -1,
        ...     "shuffle": True,
        ...     "seed": 42,
        ...     "fail_on_invalid": False,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)
    """

    name = "validated_archive"
    ConfigType = ValidatedArchiveConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Load and verify pre-computed (x, b) pairs.

        Args:
            matrix: System matrix (already normalized by orchestrator)
            cfg: Configuration dictionary
            archive: Optional pre-loaded archive data (ignored)

        Returns:
            GeneratedSamples with loaded and verified data

        Raises:
            ValueError: If globs not provided, file counts mismatch,
                       or verification fails (when fail_on_invalid=True)
            FileNotFoundError: If no files match patterns
        """
        # Validate and convert to typed config
        config = ValidatedArchiveConfig(**cfg)

        # Extract configuration
        solutions_glob = config.solutions_glob
        rhs_glob = config.rhs_glob
        samples = config.samples
        shuffle = config.shuffle
        seed = config.seed
        skip = config.skip
        verification_tolerance = config.verification_tolerance
        fail_on_invalid = config.fail_on_invalid

        logger.info(
            f"Loading (solution, RHS) pairs from archive:\n"
            f"  Solutions: {solutions_glob}\n"
            f"  RHS: {rhs_glob}"
        )

        # Layer 1: Input provision (load paired vectors from files)
        provider = PairedFileInputProvider(
            solution_glob=solutions_glob,
            rhs_glob=rhs_glob,
            shuffle=shuffle,
            seed=seed,
            skip=skip,
            file_indices=config.file_indices,
        )
        rng = np.random.default_rng(seed)
        solutions, rhs = provider.provide(matrix, count=samples, rng=rng)

        logger.info(f"Loaded {len(solutions)} (solution, RHS) pairs")

        # Layer 2: Transformation (verify consistency)
        logger.info(f"Verifying consistency (tolerance: {verification_tolerance:.2e})...")
        transform = VerifyTransform(matrix, tolerance=verification_tolerance)
        residuals = transform.transform((solutions, rhs))

        # Report results
        max_error = float(np.max(residuals))
        logger.info(f"Verification complete: max relative error = {max_error:.2e}")

        # Check for invalid pairs
        invalid_indices = np.where(residuals > verification_tolerance)[0].tolist()
        all_valid = len(invalid_indices) == 0

        if not all_valid:
            num_invalid = len(invalid_indices)
            msg = (
                f"Found {num_invalid}/{len(solutions)} invalid pairs "
                f"(max error: {max_error:.2e} > tolerance: {verification_tolerance:.2e})"
            )

            if fail_on_invalid:
                raise ValueError(
                    f"{msg}\nInvalid indices: {invalid_indices[:10]}..."
                    if len(invalid_indices) > 10
                    else f"{msg}\nInvalid indices: {invalid_indices}"
                )
            else:
                logger.warning(f"{msg}")

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs,
            solutions=solutions,
            residual_traces=None,
            error_traces=None,
        )
