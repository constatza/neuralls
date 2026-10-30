"""The single reuse decision every stage goes through."""

from __future__ import annotations

from loguru import logger

from neuralls.composition.tracking.services import TrackingServices
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


def reuse_lookup(
    tracking: TrackingServices,
    *,
    tracking_uri: str,
    experiment: str,
    identity: StageIdentity,
    force: bool,
    label: str,
    require_checkpoint: bool = False,
) -> Reused | None:
    """Build the experiment's identity store and run the reuse decision on it.

    Args:
        tracking (TrackingServices): Source of the identity store.
        tracking_uri (str): Tracking store holding the stage's runs.
        experiment (str): Experiment the stage's runs live in.
        identity (StageIdentity): The output's derived identity.
        force (bool): Skip the lookup and always execute.
        label (str): Human-readable name of the item, for the log line.
        require_checkpoint (bool): Only runs with a real checkpoint count (training).

    Returns:
        Reused | None: The reusable artifact, or ``None`` when the stage must execute.
    """
    store = tracking.identity_store(
        tracking_uri=tracking_uri, experiment=experiment, require_checkpoint=require_checkpoint
    )
    return gate_reuse(store, identity, force=force, label=label)
