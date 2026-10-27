"""Plain-text labels rendered from preconditioner descriptions.

Transitional string sink for the comparison workflow: the structured descriptions
in ``descriptions.py`` are the source of truth, and this module only renders them.
It disappears once plots and tracking consume the presentation objects directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from torchalg.preconditioners.base import Preconditioner

from neuralls.composition.preconditioners.descriptions import (
    describe_configured_preconditioner,
    describe_realized_preconditioner,
)
from neuralls.platform.config.models.preconditioner import ConcretePreconditionerConfig
from neuralls.shared.scientific_description.catalog import (
    ALGEBRAIC_DISTANCE_THRESHOLD,
    DAMPING,
    DROP_TOLERANCE,
    RANK,
    SMOOTHER_DAMPING,
    THRESHOLD,
)
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import LEGEND_PROFILE, NotationPolicy
from neuralls.shared.scientific_description.render import render

__all__ = [
    "build_configured_preconditioner_label",
    "build_preconditioner_labels",
    "contextualize_preconditioner_label",
]

_LABEL_PROFILE = replace(
    LEGEND_PROFILE,
    max_length=None,
    precision={
        THRESHOLD: 2,
        DAMPING: 2,
        SMOOTHER_DAMPING: 2,
        ALGEBRAIC_DISTANCE_THRESHOLD: 2,
        DROP_TOLERANCE: 1,
        RANK: 4,
    },
)
"""The study's legend precision; length never removes a fact, so no budget is set."""


def build_configured_preconditioner_label(
    cfg: ConcretePreconditionerConfig, *, policy: NotationPolicy | None = None
) -> str:
    """Render the complete configured description of ``cfg``."""
    description = describe_configured_preconditioner(cfg)
    return render(description, _LABEL_PROFILE, policy or NotationPolicy.default()).text


def build_preconditioner_labels(
    preconditioners: Mapping[str, Preconditioner],
    configs: Mapping[str, ConcretePreconditionerConfig],
    *,
    policy: NotationPolicy | None = None,
) -> dict[str, str]:
    """Render configured plus realized facts for each constructed preconditioner."""
    labels: dict[str, str] = {}
    for name, precond in preconditioners.items():
        configured = describe_configured_preconditioner(configs[name])
        description = ScientificDescription(
            base_name=configured.base_name,
            facts=(*configured.facts, *describe_realized_preconditioner(precond)),
        )
        labels[name] = render(description, _LABEL_PROFILE, policy or NotationPolicy.default()).text
    return labels


def contextualize_preconditioner_label(label: str, context: str | None) -> str:
    """Append the POD-2G fit dataset to a label as ``"label | context"``."""
    return label if context is None else f"{label} | {context}"
