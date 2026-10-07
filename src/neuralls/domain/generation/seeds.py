"""Deterministic, independently-labelled random seeds for generation streams."""

from __future__ import annotations

import zlib

import numpy as np


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Create random number generator from seed.

    Args:
        seed: Random seed (None for random)

    Returns:
        Random number generator
    """
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def derive_seed(mixture_seed: int, *labels: str | int) -> int:
    """Derive an independent, deterministic seed for one labelled random stream.

    Every stream that must not repeat another one (a strategy, a binding, a
    split level) takes its own label path under the mixture seed. String labels
    are hashed with CRC32, which is stable across processes unlike ``hash()``, so
    the same label path always maps to the same stream.

    Args:
        mixture_seed: Seed of the whole mixture.
        *labels: Stream discriminators, strings or non-negative integers.

    Returns:
        Non-negative integer seed for the labelled stream.
    """
    entropy = [mixture_seed, *(_label_entropy(label) for label in labels)]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _label_entropy(label: str | int) -> int:
    """Map one stream label to a SeedSequence entropy word."""
    return zlib.crc32(label.encode("utf-8")) if isinstance(label, str) else label


def derive_strategy_seed(mixture_seed: int | None, strategy_name: str) -> int | None:
    """Derive an independent, deterministic seed for one strategy in a mixture.

    Strategies in one mixture must not share a random stream: with a shared
    seed, two strategies on the same matrix emit identical RHS vectors.

    Args:
        mixture_seed: Seed of the whole mixture (None keeps non-deterministic draws).
        strategy_name: Registered strategy name used as the stream discriminator.

    Returns:
        Non-negative integer seed for the strategy, or None when mixture_seed is None.
    """
    if mixture_seed is None:
        return None
    return derive_seed(mixture_seed, strategy_name)
