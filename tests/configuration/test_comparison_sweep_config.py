"""Case-config parsing of the `[[comparison_sweeps]]` table."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.experiments import (
    AllSelector,
    CaseConfig,
    ComparisonSweepEntry,
    FractionSelector,
)


def _raw_case_config(**sweep_overrides: object) -> dict[str, object]:
    sweep: dict[str, object] = {
        "matrix_dataset": "solutions",
        "rhs_source": {"kind": "gaussian"},
        "selector": {"kind": "all"},
        "id_prefix": "sweep",
    }
    sweep.update(sweep_overrides)
    return {
        "datasets": [{"id": "solutions", "path": "datasets/solutions.toml"}],
        "comparison_sweeps": [sweep],
    }


def test_case_config_parses_comparison_sweeps_table() -> None:
    cfg = CaseConfig.model_validate(_raw_case_config())

    assert len(cfg.comparison_sweeps) == 1
    entry = cfg.comparison_sweeps[0]
    assert isinstance(entry, ComparisonSweepEntry)
    assert entry.matrix_dataset == "solutions"
    assert entry.id_prefix == "sweep"
    assert isinstance(entry.selector, AllSelector)


def test_case_config_comparison_sweeps_defaults_to_empty_list() -> None:
    cfg = CaseConfig.model_validate({"datasets": [], "jobs": []})

    assert cfg.comparison_sweeps == []


def test_case_config_parses_fraction_selector_sweep() -> None:
    cfg = CaseConfig.model_validate(
        _raw_case_config(selector={"kind": "fraction", "p": 0.25, "seed": 3})
    )

    selector = cfg.comparison_sweeps[0].selector
    assert isinstance(selector, FractionSelector)
    assert selector.p == 0.25
    assert selector.seed == 3


def test_comparison_sweep_entry_requires_id_prefix() -> None:
    sweep_without_id_prefix: dict[str, object] = {
        "matrix_dataset": "solutions",
        "rhs_source": {"kind": "gaussian"},
        "selector": {"kind": "all"},
    }
    raw = _raw_case_config()
    raw["comparison_sweeps"] = [sweep_without_id_prefix]

    with pytest.raises(ValidationError):
        CaseConfig.model_validate(raw)
