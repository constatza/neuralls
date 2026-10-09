"""Sinusoidal strategy: RHS vectors drawn as a pure 1D sinusoid over the index.

Architecture:
    Inverse mode: SinusoidalFieldInputProvider → SolveTransform
        Generates RHS vectors as sin(omega/n * i + phi) over the vector's
        own index i, with a random per-sample phase, then solves
        x = A^-1 @ b.

Every other strategy in this package draws RHS entries independently
(gaussian/uniform/etc). This strategy is the one exception: entry i is
sin(omega/n * i + phi), a smooth, non-random distribution over the vector's
own index — not drawn from the same family as the fully random strategies
above, which is the point of including it as a comparison RHS source.

omega is normalized by the vector length n (i ranges over [0, n)), so it
reads as "total radians of phase swept across the whole vector" rather than
"radians per index step" — the same omega (e.g. 2*pi*k for k full cycles)
produces a comparably-shaped waveform whether n is 100 or 10000, instead of
oscillating faster as n grows for a fixed per-step frequency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import ArchiveData, GeneratedSamples
from ..runner import register_strategy
from ..strategy_configs import SinusoidalFieldConfig
from ..transforms import SolveTransform

_TWO_PI = 2.0 * np.pi


class SinusoidalFieldInputProvider:
    """Generate sin(omega/n * i + phi) vectors, with a random phase per sample."""

    def __init__(self, *, omega: float) -> None:
        """Initialize the sinusoidal provider.

        Args:
            omega: Total phase (radians) swept across the vector in
                sin(omega/n * i + phi) — e.g. 2*pi*k for k full cycles,
                independent of the vector length n.
        """
        self.omega = omega

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate sinusoidal vectors.

        Args:
            matrix: System matrix (used only for its dimension, otherwise ignored).
            count: Number of vectors to generate.
            rng: Random number generator for per-sample phase offsets.

        Returns:
            np.ndarray: Vectors of shape (count, n), entry i is
            sin(omega/n * i + phi).
        """
        n = matrix.shape[0]
        i = np.arange(n)
        phase = rng.uniform(0.0, _TWO_PI, size=count)
        return np.sin((self.omega / n) * i[np.newaxis, :] + phase[:, np.newaxis])


@register_strategy
class SinusoidalFieldStrategy:
    """Inverse mode: generate a sin(omega * i + phi) RHS, solve for x = A^-1 @ b.

    Architecture:
        Layer 1: SinusoidalFieldInputProvider → generate RHS vectors
        Layer 2: SolveTransform → solve x = A^-1 @ b

    Use when a smooth, non-random RHS distribution is wanted for training or
    comparison data.
    """

    name = "sinusoidal_field"
    ConfigType = SinusoidalFieldConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate sinusoidal RHS vectors and solve for solutions.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary.
            archive: Optional archive data (ignored).

        Returns:
            GeneratedSamples with sinusoidal RHS vectors and solved solutions.
        """
        config = SinusoidalFieldConfig(**cfg)
        rng = np.random.default_rng(config.seed)

        # Layer 1: Input provision (sinusoidal RHS)
        provider = SinusoidalFieldInputProvider(omega=config.omega)
        rhs = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (solve for solutions)
        solutions = SolveTransform(matrix, **config.solve_config.model_dump()).transform(rhs)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)
