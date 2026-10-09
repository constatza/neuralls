"""Expanding one [[comparison_sweeps]] entry into concrete comparison entries."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.infrastructure.io.index import save_split_indices
from dlkit.infrastructure.types.split import IndexSplit

from neuralls.composition.comparison.test_set_sweep import expand_test_set_sweep
from neuralls.platform.config.models.experiments import (
    AllSelector,
    ComparisonSweepEntry,
    FirstNSelector,
    FractionSelector,
    SubsetSelector,
)


def _write_split(dataset_dir: Path, split: IndexSplit) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_split_indices(split, dataset_dir / "split.json")


def _sweep_entry(
    *, selector: object, dataset_dir: Path, id_prefix: str = "sweep"
) -> ComparisonSweepEntry:
    return ComparisonSweepEntry.model_validate(
        {
            "matrix_dataset": "solutions",
            "rhs_source": {"kind": "dataset", "path": str(dataset_dir)},
            "selector": selector,
            "id_prefix": id_prefix,
        }
    )


def test_expand_all_selector_covers_full_fold(tmp_path: Path) -> None:
    _write_split(tmp_path, IndexSplit(train=(0, 1), validation=(), test=(2, 3, 4), predict=None))
    entry = _sweep_entry(selector={"kind": "all"}, dataset_dir=tmp_path)

    expanded = expand_test_set_sweep(entry, dataset_dir=tmp_path)

    assert [e.test_position for e in expanded] == [0, 1, 2]
    assert [e.id for e in expanded] == ["sweep-0", "sweep-1", "sweep-2"]


def test_expand_first_n_selector_clamps_to_fold_length(tmp_path: Path) -> None:
    _write_split(tmp_path, IndexSplit(train=(0,), validation=(), test=(1, 2), predict=None))
    entry = _sweep_entry(selector={"kind": "first_n", "n": 10}, dataset_dir=tmp_path)

    expanded = expand_test_set_sweep(entry, dataset_dir=tmp_path)

    assert [e.test_position for e in expanded] == [0, 1]


def test_expand_fraction_selector_is_deterministic_and_matches_round_count(
    tmp_path: Path,
) -> None:
    _write_split(
        tmp_path,
        IndexSplit(train=(0,), validation=(), test=(1, 2, 3, 4, 5, 6), predict=None),
    )
    entry = _sweep_entry(selector={"kind": "fraction", "p": 0.5, "seed": 7}, dataset_dir=tmp_path)

    expanded_a = expand_test_set_sweep(entry, dataset_dir=tmp_path)
    expanded_b = expand_test_set_sweep(entry, dataset_dir=tmp_path)

    assert len(expanded_a) == round(0.5 * 6)
    assert [e.test_position for e in expanded_a] == [e.test_position for e in expanded_b]
    assert sorted(e.test_position for e in expanded_a) == [e.test_position for e in expanded_a]


def test_expand_subset_selector_passes_through_unvalidated(tmp_path: Path) -> None:
    _write_split(tmp_path, IndexSplit(train=(0,), validation=(), test=(1, 2), predict=None))
    entry = _sweep_entry(selector={"kind": "subset", "positions": [0, 99]}, dataset_dir=tmp_path)

    expanded = expand_test_set_sweep(entry, dataset_dir=tmp_path)

    assert [e.test_position for e in expanded] == [0, 99]


def test_expand_entry_copies_all_shared_fields_from_sweep_entry(tmp_path: Path) -> None:
    _write_split(tmp_path, IndexSplit(train=(0,), validation=(), test=(1,), predict=None))
    entry = ComparisonSweepEntry.model_validate(
        {
            "matrix_dataset": "solutions",
            "rhs_source": {"kind": "dataset", "path": str(tmp_path)},
            "selector": {"kind": "all"},
            "id_prefix": "sweep",
            "require_non_residual_rhs": False,
            "seed": 11,
            "assignments": ["a1"],
            "display_name": "My Sweep",
        }
    )

    expanded = expand_test_set_sweep(entry, dataset_dir=tmp_path)

    assert len(expanded) == 1
    result = expanded[0]
    assert result.matrix_dataset == entry.matrix_dataset
    assert result.rhs_source == entry.rhs_source
    assert result.require_non_residual_rhs == entry.require_non_residual_rhs
    assert result.seed == entry.seed
    assert result.assignments == entry.assignments
    assert result.display_name == entry.display_name
    assert result.test_position == 0
    assert result.id == "sweep-0"


def test_subset_selector_rejects_negative_positions_at_parse_time() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SubsetSelector(positions=[-1])


def test_fraction_selector_rejects_p_out_of_range() -> None:
    with pytest.raises(ValueError):
        FractionSelector(p=0.0, seed=1)
    with pytest.raises(ValueError):
        FractionSelector(p=1.5, seed=1)


def test_first_n_selector_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        FirstNSelector(n=0)


def test_all_selector_has_no_extra_fields() -> None:
    assert AllSelector().kind == "all"
