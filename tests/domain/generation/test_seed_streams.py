"""Labelled random streams under one mixture seed must be deterministic and distinct."""

from __future__ import annotations

import pytest

from neuralls.domain.generation.helpers import derive_seed, derive_strategy_seed

MIXTURE_SEED = 1234


def test_derive_seed_is_deterministic_for_one_label_path() -> None:
    assert derive_seed(MIXTURE_SEED, "binding", 3) == derive_seed(MIXTURE_SEED, "binding", 3)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("binding", 0), ("binding", 1)),
        (("binding", 0), ("binding-split", 0)),
        (("matrix-split",), ("binding-split", 0)),
        (("gaussian_forward",), ("uniform_forward",)),
        (("binding", 0, "gaussian_forward"), ("binding", 1, "gaussian_forward")),
    ],
)
def test_distinct_label_paths_give_distinct_streams(
    first: tuple[str | int, ...], second: tuple[str | int, ...]
) -> None:
    assert derive_seed(MIXTURE_SEED, *first) != derive_seed(MIXTURE_SEED, *second)


def test_label_order_matters() -> None:
    assert derive_seed(MIXTURE_SEED, "binding", 1) != derive_seed(MIXTURE_SEED, 1, "binding")


def test_strategy_seed_is_the_single_name_label_path() -> None:
    assert derive_strategy_seed(MIXTURE_SEED, "gaussian_forward") == derive_seed(
        MIXTURE_SEED, "gaussian_forward"
    )


def test_strategy_seed_none_stays_none() -> None:
    assert derive_strategy_seed(None, "gaussian_forward") is None
