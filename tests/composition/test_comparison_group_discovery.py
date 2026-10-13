"""Tests for data-driven comparison grouping: `_discover_comparison_groups`
and `_validate_one_matrix_shape_per_case` in `comparison_batch.py`.

Builds real npy-backed dataset directories (one single-matrix, one
multi-matrix) and hand-written `ComparisonRegistryEntry` objects — no TOML
case file needed, since these two functions only need a `CaseConfig` with a
`[[datasets]]` registry and a `config_dir` to resolve dataset paths through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.assignments.comparison_batch import (
    _discover_comparison_groups,
    _group_key_by_matrix,
    _group_key_by_matrix_and_rhs_kind,
    _validate_one_matrix_shape_per_case,
)
from neuralls.composition.comparison.models import MultiMatrixDataset, SingleMatrixDataset
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest


def _write_dataset_dir(dataset_dir: Path, *, n_matrix_samples: int) -> None:
    """Write a minimal npy-backed dataset with `n_matrix_samples` distinct matrices."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    vec_dim = 2
    matrix_arr = np.stack([np.eye(vec_dim) * (index + 1.0) for index in range(n_matrix_samples)])
    rhs_arr = np.ones((n_matrix_samples, vec_dim), dtype=np.float64)
    solutions_arr = np.ones((n_matrix_samples, vec_dim), dtype=np.float64)
    np.save(dataset_dir / "matrix.npy", matrix_arr)
    np.save(dataset_dir / "rhs.npy", rhs_arr)
    np.save(dataset_dir / "solutions.npy", solutions_arr)
    save_dataset_manifest(
        dataset_dir,
        make_dataset_manifest(
            matrix=DatasetArtifact(
                path="matrix.npy",
                format="npy",
                dtype="float64",
                shape=(n_matrix_samples, vec_dim, vec_dim),
                n_matrix_samples=n_matrix_samples,
                broadcast=False,
            ),
            rhs=DatasetArtifact(
                path="rhs.npy", format="npy", dtype="float64", shape=(n_matrix_samples, vec_dim)
            ),
            solutions=DatasetArtifact(
                path="solutions.npy",
                format="npy",
                dtype="float64",
                shape=(n_matrix_samples, vec_dim),
            ),
            normalization=DatasetNormalization(
                type="none", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
            ),
        ),
    )


def _write_dataset_registry_config(path: Path, dataset_id: str) -> None:
    path.write_text(f'id = "{dataset_id}"\n\n[output]\n', encoding="utf-8")


def _settings(tmp_path: Path) -> NeurallsSettings:
    return NeurallsSettings(
        _env_file=[], processed_dir=tmp_path / "processed", output_dir=tmp_path / "output"
    )


def _entry(
    entry_id: str, *, matrix_dataset: str, matrix_index: int = 0, rhs_kind: str = "gaussian"
) -> ComparisonRegistryEntry:
    rhs_source = {"kind": rhs_kind}
    if rhs_kind == "sinusoidal":
        rhs_source["omega"] = 6.283185307179586
    return ComparisonRegistryEntry(
        id=entry_id,
        matrix_dataset=matrix_dataset,
        matrix_index=matrix_index,
        rhs_source=rhs_source,
    )


@pytest.fixture
def case_setup(tmp_path: Path) -> tuple[CaseConfig, Path, NeurallsSettings]:
    """One single-matrix dataset ('single') and one multi-matrix dataset ('multi')."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_dataset_registry_config(config_dir / "single.toml", "single")
    _write_dataset_registry_config(config_dir / "multi.toml", "multi")

    settings = _settings(tmp_path)
    _write_dataset_dir(settings.processed_dir / "single", n_matrix_samples=1)
    _write_dataset_dir(settings.processed_dir / "multi", n_matrix_samples=3)

    master_cfg = CaseConfig.model_validate(
        {
            "datasets": [
                {"id": "single", "path": "single.toml"},
                {"id": "multi", "path": "multi.toml"},
            ],
            "comparisons": [],
        }
    )
    return master_cfg, config_dir, settings


def test_single_matrix_dataset_resolves_to_single_matrix_identity(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [_entry("a", matrix_dataset="single"), _entry("b", matrix_dataset="single")]

    groups = _discover_comparison_groups(
        entries, master_cfg, config_dir, settings, key_fn=_group_key_by_matrix_and_rhs_kind
    )

    assert len(groups) == 1
    (group,) = groups.values()
    assert isinstance(group.matrix_identity, SingleMatrixDataset)
    assert group.entry_ids == frozenset({"a", "b"})


def test_hand_written_entries_sharing_dataset_and_kind_group_together(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Two plain entries, no sweep involved — grouped purely by what they test."""
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("eval-0", matrix_dataset="multi", matrix_index=0),
        _entry("eval-1", matrix_dataset="multi", matrix_index=1),
    ]

    groups = _discover_comparison_groups(
        entries, master_cfg, config_dir, settings, key_fn=_group_key_by_matrix_and_rhs_kind
    )

    assert len(groups) == 1
    (group,) = groups.values()
    assert group.entry_ids == frozenset({"eval-0", "eval-1"})


def test_different_matrix_indices_resolve_to_multi_matrix_with_both_ids(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("eval-0", matrix_dataset="multi", matrix_index=0),
        _entry("eval-1", matrix_dataset="multi", matrix_index=1),
    ]

    groups = _discover_comparison_groups(
        entries, master_cfg, config_dir, settings, key_fn=_group_key_by_matrix_and_rhs_kind
    )

    (group,) = groups.values()
    assert isinstance(group.matrix_identity, MultiMatrixDataset)
    assert group.matrix_identity.resolved_indices == frozenset({0, 1})


def test_same_matrix_index_within_multi_matrix_dataset_still_resolves_multi(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Both entries happen to pick index 0 — still a MultiMatrixDataset (the
    dataset itself has several matrices), just with one resolved id."""
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("eval-0a", matrix_dataset="multi", matrix_index=0),
        _entry("eval-0b", matrix_dataset="multi", matrix_index=0),
    ]

    groups = _discover_comparison_groups(
        entries, master_cfg, config_dir, settings, key_fn=_group_key_by_matrix_and_rhs_kind
    )

    (group,) = groups.values()
    assert isinstance(group.matrix_identity, MultiMatrixDataset)
    assert group.matrix_identity.resolved_indices == frozenset({0})


def test_different_rhs_kinds_split_apart_under_fine_grouping(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Same matrix, different rhs_source.kind — the fine grouping keeps them apart."""
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("gaussian-run", matrix_dataset="single", rhs_kind="gaussian"),
        _entry("sinusoidal-run", matrix_dataset="single", rhs_kind="sinusoidal"),
    ]

    groups = _discover_comparison_groups(
        entries, master_cfg, config_dir, settings, key_fn=_group_key_by_matrix_and_rhs_kind
    )

    assert len(groups) == 2
    assert {group.entry_ids for group in groups.values()} == {
        frozenset({"gaussian-run"}),
        frozenset({"sinusoidal-run"}),
    }


def test_different_rhs_kinds_pool_together_under_coarse_grouping(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Same matrix, different rhs_source.kind — the coarse grouping pools them,
    since rhs_source.kind has no bearing on whether setup() can be reused."""
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("gaussian-run", matrix_dataset="single", rhs_kind="gaussian"),
        _entry("sinusoidal-run", matrix_dataset="single", rhs_kind="sinusoidal"),
    ]

    groups = _discover_comparison_groups(
        entries, master_cfg, config_dir, settings, key_fn=_group_key_by_matrix
    )

    assert len(groups) == 1
    (group,) = groups.values()
    assert group.entry_ids == frozenset({"gaussian-run", "sinusoidal-run"})


def test_case_mixing_single_and_multi_matrix_datasets_is_rejected(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("a", matrix_dataset="single"),
        _entry("b", matrix_dataset="multi"),
    ]

    with pytest.raises(ValueError, match="mixes single-matrix and multi-matrix"):
        _validate_one_matrix_shape_per_case(entries, master_cfg, config_dir, settings)


def test_case_with_one_matrix_shape_passes_validation(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("eval-0", matrix_dataset="multi", matrix_index=0),
        _entry("eval-1", matrix_dataset="multi", matrix_index=1),
    ]

    _validate_one_matrix_shape_per_case(entries, master_cfg, config_dir, settings)  # no raise
