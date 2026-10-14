"""Case-config parsing of the `[[comparison_sweeps]]` table."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonSweepEntry,
    FirstNSelector,
    FractionSelector,
)


def _raw_case_config(**sweep_overrides: object) -> dict[str, object]:
    sweep: dict[str, object] = {
        "matrix_dataset": "solutions",
        "rhs_source": {"kind": "gaussian"},
        "selector": {"kind": "first_n", "n": 2},
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
    assert isinstance(entry.selector, FirstNSelector)
    assert entry.selector.n == 2


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
        "selector": {"kind": "first_n", "n": 2},
    }
    raw = _raw_case_config()
    raw["comparison_sweeps"] = [sweep_without_id_prefix]

    with pytest.raises(ValidationError):
        CaseConfig.model_validate(raw)


@pytest.mark.parametrize(
    "selector",
    [
        {"kind": "all"},
        {"kind": "subset", "positions": [0, 1]},
        {"kind": "fraction", "p": 1.5, "seed": 0},
        {"kind": "fraction", "p": 0.0, "seed": 0},
    ],
)
def test_comparison_sweep_rejects_selectors_that_are_not_a_count_or_ratio(
    selector: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CaseConfig.model_validate(_raw_case_config(selector=selector))


def test_comparison_sweep_full_fraction_is_every_held_out_matrix() -> None:
    cfg = CaseConfig.model_validate(
        _raw_case_config(selector={"kind": "fraction", "p": 1.0, "seed": 0})
    )

    assert cfg.comparison_sweeps[0].selector == FractionSelector(p=1.0, seed=0)


def test_rhs_count_defaults_to_one_and_must_be_positive() -> None:
    assert CaseConfig.model_validate(_raw_case_config()).comparison_sweeps[0].rhs_count == 1
    assert (
        CaseConfig.model_validate(_raw_case_config(rhs_count=5)).comparison_sweeps[0].rhs_count == 5
    )
    with pytest.raises(ValidationError):
        CaseConfig.model_validate(_raw_case_config(rhs_count=0))


@pytest.mark.parametrize("removed_field", ["matrix_index", "test_position"])
def test_comparison_entry_rejects_removed_matrix_selection_fields(removed_field: str) -> None:
    raw = {
        "datasets": [{"id": "solutions", "path": "datasets/solutions.toml"}],
        "comparisons": [
            {
                "id": "c",
                "matrix_dataset": "solutions",
                "rhs_source": {"kind": "gaussian"},
                removed_field: 0,
            }
        ],
    }

    with pytest.raises(ValidationError):
        CaseConfig.model_validate(raw)
