"""Family classification for preconditioner configs.

Groups preconditioner configs for consumers that need to treat related
variants alike (today: comparison-plot styling) without those consumers
reaching into config internals themselves.
"""

from __future__ import annotations

from neuralls.platform.config.models.coarsening import (
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
)
from neuralls.shared.types import PreconditionerFamily

__all__ = ["PreconditionerFamilyKey", "pod2g_style_keys", "preconditioner_family"]

type PreconditionerFamilyKey = PreconditionerFamily | PreconditionerType
"""A preconditioner's family: one of the merged `PreconditionerFamily` names
(`amg`/`pod2g`/`neural`), or, for every other preconditioner type, that
type's own `PreconditionerType` member directly — no separate family name
needed since each already denotes a distinct group."""


def preconditioner_family(cfg: PreconditionerConfig) -> PreconditionerFamilyKey:
    """Classify a preconditioner config into a family.

    AMG splits into ``PreconditionerFamily.POD2G`` (POD-2G coarsening,
    inline-fit or checkpoint-predicted) and ``PreconditionerFamily.AMG``
    (classical aggregation/target-dim coarsening): the two are routinely
    compared side by side but are architecturally distinct methods.
    ``NEURAL`` and ``NEURAL_AMG`` collapse to a single
    ``PreconditionerFamily.NEURAL``. Every other type is its own family —
    its own ``PreconditionerType`` member, unmodified — since each already
    appears at most once in a typical comparison, so no further grouping is
    needed.

    Args:
        cfg (PreconditionerConfig): Preconditioner config, as constructed by
            the case TOML loader.

    Returns:
        PreconditionerFamilyKey: Family key for this config.
    """
    if isinstance(cfg, AMGPreconditionerConfig):
        if isinstance(cfg.coarsening, (PODCoarseningConfig, NeuralPODCoarseningConfig)):
            return PreconditionerFamily.POD2G
        return PreconditionerFamily.AMG
    if cfg.type in (PreconditionerType.NEURAL, PreconditionerType.NEURAL_AMG):
        return PreconditionerFamily.NEURAL
    return cfg.type


def pod2g_style_keys(cfg: PreconditionerConfig) -> tuple[str | None, str | None]:
    """Extract (color_key, marker_key) for a POD-2G config, else (None, None).

    POD-2G comparisons routinely sweep many (fit dataset, snapshot weighting)
    combinations in one plot; the fit dataset drives color, the weighting
    scheme drives marker, so a dense sweep separates into distinct
    color/marker combinations instead of collapsing onto one family-wide
    marker+linestyle with only color-shade differences.

    Args:
        cfg: Preconditioner configuration to inspect.

    Returns:
        tuple[str | None, str | None]: ``(color_key, marker_key)``, both
        ``None`` for non-POD-2G configs (falls back to family-based styling).
    """
    if not isinstance(cfg, AMGPreconditionerConfig) or not isinstance(
        cfg.coarsening, PODCoarseningConfig | NeuralPODCoarseningConfig
    ):
        return None, None
    color_key = str(cfg.coarsening.dataset_dir)
    # Only PODCoarseningConfig carries a snapshot weighting scheme;
    # NeuralPODCoarseningConfig predicts snapshots from a checkpoint instead.
    marker_key = (
        cfg.coarsening.weighting.method if isinstance(cfg.coarsening, PODCoarseningConfig) else None
    )
    return color_key, marker_key
