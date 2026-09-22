"""The single reuse decision every stage goes through."""

from __future__ import annotations

from loguru import logger

from neuralls.domain.identity import Reused, StageIdentity
from neuralls.domain.identity_ports import IdentityStore


def gate_reuse(
    store: IdentityStore, identity: StageIdentity, *, force: bool, label: str
) -> Reused | None:
    """Decide whether a stage's output can be reused, and say why.

    The only place that combines ``force`` with the identity lookup, so every
    stage handles it identically and every decision is logged the same way.

    Args:
        store (IdentityStore): Where previously produced artifacts are found.
        identity (StageIdentity): The output's derived identity.
        force (bool): Skip the lookup and always execute.
        label (str): Human-readable name of the item, for the log line.

    Returns:
        Reused | None: The reusable artifact, or ``None`` when the stage must execute.
    """
    short = identity.key[:19]
    if force:
        logger.info("{} '{}': executing (forced)", identity.stage, label)
        return None
    reused = store.find(identity)
    if reused is None:
        logger.info(
            "{} '{}': executing — no prior run with identity {} (inputs: {})",
            identity.stage,
            label,
            short,
            dict(identity.components),
        )
        return None
    logger.info(
        "{} '{}': reusing run {} (identity {})", identity.stage, label, reused.run_id, short
    )
    return reused
