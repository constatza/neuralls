"""Every registered strategy config is described, or explicitly described as parameterless."""

from __future__ import annotations

from importlib import import_module

import pytest
from pydantic import BaseModel

from neuralls.domain.generation.descriptions import (
    UndescribedStrategyConfigError,
    _parameter_facts,
)
from neuralls.domain.generation.runner import registered_config_type, registered_strategy_names
from neuralls.domain.generation.strategy_configs import BaseStrategyConfig

import_module("neuralls.domain.generation.strategies")  # registers every strategy

_REQUIRED_FIELDS: dict[str, dict[str, object]] = {
    "residuals": {"stop": 5},
    "gaussian_residuals": {"stop": 5},
    "smoother_filtered_probes": {"stop": 5},
    "rhs_archive": {"rhs_glob": "rhs/*.npy"},
    "scaled_solutions": {"solutions_glob": "x/*.npy"},
    "solution_archive": {"solutions_glob": "x/*.npy"},
    "sparse_rhs": {"indices": [0], "values": [1.0]},
    "validated_archive": {"solutions_glob": "x/*.npy", "rhs_glob": "rhs/*.npy"},
}
_PARAMETERLESS = frozenset({"neutral_ones", "solution_archive"})


@pytest.mark.parametrize("name", registered_strategy_names())
def test_every_registered_strategy_config_is_described(name: str) -> None:
    config = registered_config_type(name).model_validate(
        {"samples": 3, **_REQUIRED_FIELDS.get(name, {})}
    )

    facts = _parameter_facts(config)

    assert (facts == ()) is (name in _PARAMETERLESS)


def test_registry_holds_the_known_strategies() -> None:
    assert {"neutral_ones", "krylov", "solution_archive"} <= set(registered_strategy_names())


def test_an_undescribed_strategy_config_fails() -> None:
    class UnheardOfConfig(BaseStrategyConfig):
        pass

    assert issubclass(UnheardOfConfig, BaseModel)
    with pytest.raises(UndescribedStrategyConfigError, match="UnheardOfConfig"):
        _parameter_facts(UnheardOfConfig(samples=1))
