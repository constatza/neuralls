"""Bridge from `SnapshotWeightingConfig` to torchalg's POD row-scale functions.

All the actual weighting math (what a row scale means, how it's computed)
lives in `torchalg.preconditioners.implementations.pod.weighting` — a
generic, `neuralls`-agnostic capability of the POD implementation itself.
This module's only job is picking which torchalg function a given TOML
config asks for; see `docs/plan.md` for why the split lands here rather than
in `domain/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from neuralls.platform.config.models.preconditioner import (
    PowerNormWeightingConfig,
    RawWeightingConfig,
    SmootherPersistenceWeightingConfig,
)

if TYPE_CHECKING:
    import torch

    from neuralls.platform.config.models.preconditioner import SnapshotWeightingConfig


def weighting_needs_matrix(cfg: SnapshotWeightingConfig) -> bool:
    """Whether resolving ``cfg`` requires the system matrix A.

    Lets callers that source snapshots and the matrix from different places
    (e.g. a dataloader that must be told to also collect the matrix feature)
    skip that extra work for `RawWeightingConfig`/`PowerNormWeightingConfig(metric="l2")`.

    Args:
        cfg: The weighting configuration to inspect.

    Returns:
        bool: `True` for `metric="a"` power-norm weighting or
            smoother-persistence weighting; `False` otherwise.
    """
    match cfg:
        case PowerNormWeightingConfig(metric="a"):
            return True
        case SmootherPersistenceWeightingConfig():
            return True
        case _:
            return False


def resolve_row_scales(
    cfg: SnapshotWeightingConfig,
    snapshots: torch.Tensor,
    matrix: torch.Tensor | None,
) -> torch.Tensor | None:
    """Compute the per-snapshot row scale a POD fit should apply, per `cfg`.

    Args:
        cfg: The weighting configuration selected in TOML.
        snapshots: Snapshot ensemble, shape (n_samples, n_dofs).
        matrix: System matrix A, required when `weighting_needs_matrix(cfg)`.

    Returns:
        torch.Tensor | None: Row scales to pass as `PODCoarseningStrategy.fit`'s
            `row_scales` argument, or `None` for `RawWeightingConfig` (today's
            unweighted fit, unchanged).

    Raises:
        ValueError: If `cfg` needs the system matrix and `matrix` is `None`.
    """
    from torchalg.preconditioners.implementations.pod import (
        power_norm_scales,
        smoother_persistence_scales,
    )

    if weighting_needs_matrix(cfg) and matrix is None:
        raise ValueError(f"{type(cfg).__name__} requires the system matrix, but none was given.")

    match cfg:
        case RawWeightingConfig():
            return None
        case PowerNormWeightingConfig(metric=metric, beta=beta):
            return power_norm_scales(snapshots, matrix=matrix, metric=metric, beta=beta)
        case SmootherPersistenceWeightingConfig(omega=omega, steps=steps):
            return smoother_persistence_scales(
                snapshots, cast("torch.Tensor", matrix), omega=omega, steps=steps
            )
