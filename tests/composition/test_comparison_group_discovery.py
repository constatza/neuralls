"""Tests for data-driven comparison grouping: `discover_comparison_groupings`,
`validate_one_matrix_shape_per_case` and `validate_group_specs_consistent`
in `composition/comparison/groups.py`.

Builds real npy-backed dataset directories (one single-matrix, one
multi-matrix) and hand-written `ComparisonRegistryEntry` objects — no TOML
case file needed, since these two functions only need a `CaseConfig` with a
`[[datasets]]` registry and a `config_dir` to resolve dataset paths through.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison.groups import (
    discover_comparison_groupings,
    validate_group_specs_consistent,
    validate_one_matrix_shape_per_case,
)
from neuralls.composition.comparison.models import (
    ComparisonGroup,
    GroupKey,
    MultiMatrixDataset,
    SingleMatrixDataset,
)
from neuralls.composition.identity.comparison import preconditioner_identities
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.shared.digest import Digest
from neuralls.shared.types import GroupId


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

    groups = discover_comparison_groupings(
        entries, master_cfg, config_dir, settings
    ).by_matrix_and_rhs_kind

    assert len(groups) == 1
    (group,) = groups.values()
    assert isinstance(group.matrix_identity, SingleMatrixDataset)
    assert group.entry_ids == ("a", "b")


def test_hand_written_entries_sharing_dataset_and_kind_group_together(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Two plain entries, no sweep involved — grouped purely by what they test."""
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("eval-0", matrix_dataset="multi", matrix_index=0),
        _entry("eval-1", matrix_dataset="multi", matrix_index=1),
    ]

    groups = discover_comparison_groupings(
        entries, master_cfg, config_dir, settings
    ).by_matrix_and_rhs_kind

    assert len(groups) == 1
    (group,) = groups.values()
    assert group.entry_ids == ("eval-0", "eval-1")


def test_different_matrix_indices_resolve_to_multi_matrix_with_both_ids(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("eval-0", matrix_dataset="multi", matrix_index=0),
        _entry("eval-1", matrix_dataset="multi", matrix_index=1),
    ]

    groups = discover_comparison_groupings(
        entries, master_cfg, config_dir, settings
    ).by_matrix_and_rhs_kind

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

    groups = discover_comparison_groupings(
        entries, master_cfg, config_dir, settings
    ).by_matrix_and_rhs_kind

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

    groups = discover_comparison_groupings(
        entries, master_cfg, config_dir, settings
    ).by_matrix_and_rhs_kind

    assert len(groups) == 2
    assert {group.entry_ids for group in groups.values()} == {
        ("gaussian-run",),
        ("sinusoidal-run",),
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

    groups = discover_comparison_groupings(entries, master_cfg, config_dir, settings).by_matrix

    assert len(groups) == 1
    (group,) = groups.values()
    assert group.entry_ids == ("gaussian-run", "sinusoidal-run")


def test_case_mixing_single_and_multi_matrix_datasets_is_rejected(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("a", matrix_dataset="single"),
        _entry("b", matrix_dataset="multi"),
    ]

    with pytest.raises(ValueError, match="mixes single-matrix and multi-matrix"):
        validate_one_matrix_shape_per_case(entries, master_cfg, config_dir, settings)


def test_case_with_one_matrix_shape_passes_validation(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("eval-0", matrix_dataset="multi", matrix_index=0),
        _entry("eval-1", matrix_dataset="multi", matrix_index=1),
    ]

    validate_one_matrix_shape_per_case(entries, master_cfg, config_dir, settings)  # no raise


def _jacobi(**overrides: int) -> StandardPreconditionerConfig:
    """Jacobi's derived name ignores schedule fields, so `start_iter` yields a same-name conflict."""
    return StandardPreconditionerConfig(type=PreconditionerType.JACOBI, **overrides)


def test_group_members_keep_declared_config_order(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Order is config order, never hash order, so "first member" is reproducible."""
    master_cfg, config_dir, settings = case_setup
    ids = ["z-last-alphabetically-first", "m", "a-first-alphabetically-last"]
    entries = [
        _entry(entry_id, matrix_dataset="multi", matrix_index=i) for i, entry_id in enumerate(ids)
    ]

    (group,) = discover_comparison_groupings(
        entries, master_cfg, config_dir, settings
    ).by_matrix.values()

    assert group.entry_ids == tuple(ids)


def test_group_key_renders_once_and_coarse_key_owns_the_cache(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [
        _entry("g", matrix_dataset="single", rhs_kind="gaussian"),
        _entry("s", matrix_dataset="single", rhs_kind="sinusoidal"),
    ]

    groupings = discover_comparison_groupings(entries, master_cfg, config_dir, settings)

    fine = groupings.by_matrix_and_rhs_kind["single::gaussian"]
    assert fine.key == GroupKey(dataset_id="single", rhs_kind="gaussian")
    coarse = groupings.cache_scope(fine)
    assert coarse is groupings.by_matrix["single"]
    assert groupings.group_for_entry("s") is coarse
    assert dict(coarse.rhs_kind_counts) == {"gaussian": 1, "sinusoidal": 1}


def _identities(spec: PreconditionerConfig, *digests: str) -> dict[str, Digest]:
    """Identity of one spec whose checkpoint (if any) has the given content digests."""
    return preconditioner_identities([spec], {f"0:ckpt{i}": d for i, d in enumerate(digests)})


def _two_entry_groups(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> Mapping[GroupId, ComparisonGroup]:
    master_cfg, config_dir, settings = case_setup
    entries = [_entry("a", matrix_dataset="single"), _entry("b", matrix_dataset="single")]
    return discover_comparison_groupings(entries, master_cfg, config_dir, settings).by_matrix


def test_group_whose_members_define_one_name_differently_is_rejected(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    groups = _two_entry_groups(case_setup)

    with pytest.raises(ValueError, match="configured differently in entries 'a' and 'b'"):
        validate_group_specs_consistent(
            groups, {"a": _identities(_jacobi()), "b": _identities(_jacobi(start_iter=5))}
        )


def test_same_neural_name_with_different_checkpoint_content_is_rejected(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Both are `neural_assignment_pod_run`; only the checkpoint bytes differ."""
    groups = _two_entry_groups(case_setup)
    first = NeuralPreconditionerConfig(assignment="pod-run")
    second = NeuralPreconditionerConfig(assignment="pod-run")
    assert first.name == second.name

    with pytest.raises(ValueError, match="different checkpoint"):
        validate_group_specs_consistent(
            groups,
            {"a": _identities(first, "sha256:aaa"), "b": _identities(second, "sha256:bbb")},
        )


def test_same_neural_model_resolved_to_different_local_paths_is_accepted(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    """Same bytes at two temp paths is one preconditioner — no false conflict."""
    groups = _two_entry_groups(case_setup)
    at_a = NeuralPreconditionerConfig(
        assignment="pod-run", resolved_checkpoint_path=Path("/tmp/a/model.ckpt")
    )
    at_b = NeuralPreconditionerConfig(
        assignment="pod-run", resolved_checkpoint_path=Path("/tmp/b/model.ckpt")
    )
    assert at_a != at_b  # why `==` was the wrong comparison

    validate_group_specs_consistent(
        groups, {"a": _identities(at_a, "sha256:same"), "b": _identities(at_b, "sha256:same")}
    )


def test_group_with_identical_specs_or_cache_hit_members_passes(
    case_setup: tuple[CaseConfig, Path, NeurallsSettings],
) -> None:
    master_cfg, config_dir, settings = case_setup
    entries = [_entry("a", matrix_dataset="single"), _entry("b", matrix_dataset="single")]
    groups = discover_comparison_groupings(entries, master_cfg, config_dir, settings).by_matrix

    validate_group_specs_consistent(groups, {"a": _identities(_jacobi())})
