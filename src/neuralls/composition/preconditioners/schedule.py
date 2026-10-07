"""Preconditioner scheduling: switch from a primary preconditioner to a fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import Identity
from torchalg.preconditioners.implementations.scheduled import ScheduledPreconditioner

from neuralls.platform.config.models.preconditioner import PreconditionerType

if TYPE_CHECKING:
    from neuralls.platform.config.models.preconditioner import ConcretePreconditionerConfig


@dataclass(frozen=True)
class PreconditionerScheduleConfig:
    """Scheduling parameters for preconditioner switching.

    Extracted from BasePreconditionerConfig for internal use.
    Separates scheduling concerns from preconditioner configuration.

    Attributes:
        start_iter: Iteration at which the primary preconditioner becomes active.
        limit_iters: Number of iterations to apply primary preconditioner.
                     -1 means unlimited (use primary for entire solve).
        fallback: Preconditioner type to switch to after limit is reached.
    """

    start_iter: int = 0
    limit_iters: int = -1
    fallback: PreconditionerType = PreconditionerType.IDENTITY


def _extract_schedule(cfg: ConcretePreconditionerConfig) -> PreconditionerScheduleConfig:
    """Extract scheduling parameters from preconditioner config.

    Pure function to extract scheduling concerns from mixed config.

    Args:
        cfg: Preconditioner configuration from TOML

    Returns:
        Extracted schedule configuration
    """
    return PreconditionerScheduleConfig(
        start_iter=cfg.start_iter,
        limit_iters=cfg.limit_iters,
        fallback=cfg.fallback,
    )


def create_scheduled_preconditioner(
    primary: Preconditioner,
    schedule: PreconditionerScheduleConfig,
    matrix: torch.Tensor | None = None,
) -> Preconditioner:
    """Create a scheduled preconditioner based on schedule config.

    ``ScheduledPreconditioner`` requires its own ``setup()`` call before
    ``apply()`` (it sets up both primary and fallback together, since both
    branches must be ready for dispatch) — when wrapping occurs, ``matrix``
    is required, and this function makes the wrapper ready before returning
    it. ``primary`` is expected to already be setup (`create_preconditioner`
    returns a ready preconditioner), so this only re-runs the fallback's
    setup plus a cheap no-op re-setup of the already-ready primary.

    Args:
        primary: Main preconditioner to apply (already setup).
        schedule: Schedule configuration with activation, limit, and fallback type.
        matrix: System matrix, required only when wrapping actually occurs.

    Returns:
        ScheduledPreconditioner (ready for ``apply()``) if delayed or
        limited, otherwise primary unchanged.

    Raises:
        ValueError: If wrapping is required but ``matrix`` was not supplied.

    Example:
        >>> # Limit neural preconditioner to first 10 iterations
        >>> schedule = PreconditionerScheduleConfig(limit_iters=10)
        >>> scheduled = create_scheduled_preconditioner(neural_precond, schedule, matrix)
    """
    if schedule.start_iter == 0 and schedule.limit_iters < 0:
        return primary

    if matrix is None:
        raise ValueError("create_scheduled_preconditioner requires `matrix` when wrapping.")

    # Create fallback preconditioner based on type
    if schedule.fallback == PreconditionerType.IDENTITY:
        fallback_precond = Identity()
    else:
        raise ValueError(f"Unsupported fallback type: {schedule.fallback}")

    return ScheduledPreconditioner(
        primary=primary,
        fallback=fallback_precond,
        limit_iters=None if schedule.limit_iters < 0 else schedule.limit_iters,
        start_iter=schedule.start_iter,
    ).setup(matrix)
