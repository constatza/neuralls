"""Direct RHS generators for comparison workflows."""

from __future__ import annotations

import numpy as np

from neuralls.platform.config.models.comparison import (
    GaussianRhsSourceModel,
    SinusoidalRhsSourceModel,
)

_TWO_PI = 2.0 * np.pi


def generate_gaussian_rhs(
    dim: int,
    cfg: GaussianRhsSourceModel,
    seed: int | None,
) -> np.ndarray:
    """Generate a direct Gaussian RHS vector."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=cfg.mean, scale=cfg.std, size=dim).astype(np.float64)


def generate_sinusoidal_rhs(
    dim: int,
    cfg: SinusoidalRhsSourceModel,
    seed: int | None,
) -> np.ndarray:
    """Generate a direct sinusoidal RHS vector: sin(omega/dim * i + phi).

    omega is normalized by dim so it reads as "total radians swept across
    the vector" (e.g. 2*pi*k for k full cycles), independent of system size —
    same convention as the sinusoidal_field generation strategy.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(dim)
    phase = rng.uniform(0.0, _TWO_PI)
    return np.sin((cfg.omega / dim) * i + phase).astype(np.float64)
