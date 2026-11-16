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

__all__ = ["PreconditionerFamilyKey", "pod_color_key", "preconditioner_family"]

type PreconditionerFamilyKey = PreconditionerFamily | PreconditionerType
"""A preconditioner's family: one of the merged `PreconditionerFamily` names
(`amg`/`pod`/`neural`), or, for every other preconditioner type, that
type's own `PreconditionerType` member directly — no separate family name
needed since each already denotes a distinct group."""


def preconditioner_family(cfg: PreconditionerConfig) -> PreconditionerFamilyKey:
    """Classify a preconditioner config into a family.

    AMG splits into ``PreconditionerFamily.POD`` (POD coarsening,
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
            return PreconditionerFamily.POD
        return PreconditionerFamily.AMG
    if cfg.type in (PreconditionerType.NEURAL, PreconditionerType.NEURAL_AMG):
        return PreconditionerFamily.NEURAL
    return cfg.type


def pod_color_key(cfg: PreconditionerConfig) -> str | None:
    """The plot colour key of a POD config: its fit dataset; ``None`` for any other config.

    POD comparisons routinely sweep many fit datasets in one plot; keying colour on the
    dataset keeps same-dataset variants one colour instead of shading one family-wide
    colour. ``None`` falls back to family-based styling.

    Args:
        cfg: Preconditioner configuration to inspect.

    Returns:
        str | None: The POD fit dataset directory, or ``None`` for non-POD configs.
    """
    if not isinstance(cfg, AMGPreconditionerConfig) or not isinstance(
        cfg.coarsening, PODCoarseningConfig | NeuralPODCoarseningConfig
    ):
        return None
    return str(cfg.coarsening.dataset_dir)
