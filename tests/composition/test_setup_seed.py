"""A randomized setup draws its seed per matrix, the same one for every build of that matrix,
from a stream apart from the seeds of the matrix's RHS draws."""

from __future__ import annotations

import pytest

from neuralls.composition.comparison._preconditioner_setup import setup_config_for_matrix
from neuralls.composition.comparison.plan import derive_seed
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.shared.types import MatrixId

CONFIGURED_SEED = 7

type SeededConfig = AdaptiveSAPreconditionerConfig | BootstrapAMGPreconditionerConfig


def _setup_seed(config: SeededConfig, matrix: int) -> int:
    built: PreconditionerConfig = setup_config_for_matrix(config, MatrixId(matrix))
    match built:
        case AdaptiveSAPreconditionerConfig() | BootstrapAMGPreconditionerConfig():
            return built.seed
        case other:
            raise AssertionError(f"a seeded config became {type(other).__name__}")


@pytest.mark.parametrize(
    "config",
    [
        AdaptiveSAPreconditionerConfig(seed=CONFIGURED_SEED),
        BootstrapAMGPreconditionerConfig(seed=CONFIGURED_SEED),
    ],
)
def test_setup_seed_differs_by_matrix_and_repeats_within_one(config: SeededConfig) -> None:
    first = _setup_seed(config, 3)

    assert first == _setup_seed(config, 3)
    assert first != _setup_seed(config, 4)
    assert first != derive_seed(CONFIGURED_SEED, MatrixId(3), 0)
    assert config.seed == CONFIGURED_SEED
