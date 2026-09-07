"""Tests for the pure helpers in ``scripts/expand_dataset_sweep.py``.

The golden-file tests double as the correctness proof that the generic
renderer reproduces the real, hand-written dataset configs under
``configs/datasets/train/rectangular-high-condition/`` exactly — not just
something similar.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAMILY_DIR = _REPO_ROOT / "configs" / "datasets" / "train" / "rectangular-high-condition"


def test_render_dataset_toml_matches_committed_gaussian_residuals_file(
    dataset_sweep_module: ModuleType,
) -> None:
    """A gaussian_residuals-shaped render matches the real gaussian-cg50-1000.toml exactly."""
    content = dataset_sweep_module._render_dataset_toml(
        id_="gaussian-cg50-rectangular-high-condition-1000",
        source={"matrix_path": "${NEURALLS_RAW_DIR}/matrices/rectangular-high-condition.txt"},
        generation_header={"normalize": "matrix", "shuffle": True, "seed": 42},
        strategy={"name": "gaussian_residuals", "samples": 1000, "cg_iters": 50},
        strategy_comment=None,
        output={"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    )
    assert content == (_FAMILY_DIR / "gaussian-cg50-1000.toml").read_text()


def test_render_dataset_toml_matches_committed_gaussian_forward_file(
    dataset_sweep_module: ModuleType,
) -> None:
    """A gaussian_forward-shaped render with strategy_comment matches gaussian-0cg-1000.toml exactly."""
    comment = [
        '"0-CG": cg_iters on the residuals/trace strategies has a hard',
        "floor of 1 (and even cg_iters=1 emits 2 rows per system - iteration 0 AND",
        "iteration 1, not just the untraced solution). gaussian_forward is the true",
        "0-CG case - x ~ N(mu, sigma), b = A @ x, no CG solve at all.",
        "POD-2G fitting never sees real archived solutions - those are comparison-only.",
    ]
    content = dataset_sweep_module._render_dataset_toml(
        id_="gaussian-0cg-rectangular-high-condition-1000",
        source={"matrix_path": "${NEURALLS_RAW_DIR}/matrices/rectangular-high-condition.txt"},
        generation_header={"normalize": "matrix", "shuffle": True, "seed": 42},
        strategy={"name": "gaussian_forward", "samples": 1000},
        strategy_comment=comment,
        output={"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    )
    assert content == (_FAMILY_DIR / "gaussian-0cg-1000.toml").read_text()


def test_find_swept_field_rejects_no_list_valued_field(dataset_sweep_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="no list-valued field"):
        dataset_sweep_module._find_swept_field({"name": "gaussian_residuals", "samples": 500})


def test_find_swept_field_rejects_multiple_list_valued_fields(
    dataset_sweep_module: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="only one field may be swept"):
        dataset_sweep_module._find_swept_field(
            {"name": "gaussian_residuals", "samples": [1, 2], "cg_iters": [10, 20]}
        )


def test_expand_sweep_rejects_wrong_strategy_count(
    dataset_sweep_module: ModuleType, tmp_path: Path
) -> None:
    spec = {
        "id": "base-id",
        "source": {"matrix_path": "m.txt"},
        "generation": {"normalize": "matrix", "shuffle": True, "seed": 1, "strategy": []},
        "output": {"data_dir": "out"},
    }
    with pytest.raises(ValueError, match="exactly one"):
        dataset_sweep_module.expand_sweep(spec, output_dir=tmp_path, filename_stem="base")


def test_expand_sweep_writes_one_file_per_value(
    dataset_sweep_module: ModuleType, tmp_path: Path
) -> None:
    spec = {
        "id": "gaussian-cg50-rectangular-high-condition",
        "source": {"matrix_path": "${NEURALLS_RAW_DIR}/matrices/rectangular-high-condition.txt"},
        "generation": {
            "normalize": "matrix",
            "shuffle": True,
            "seed": 42,
            "strategy": [{"name": "gaussian_residuals", "samples": [1000, 5000], "cg_iters": 50}],
        },
        "output": {"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    }
    written = dataset_sweep_module.expand_sweep(
        spec, output_dir=tmp_path, filename_stem="gaussian-cg50"
    )
    assert [p.name for p in written] == ["gaussian-cg50-1000.toml", "gaussian-cg50-5000.toml"]
    assert (tmp_path / "gaussian-cg50-1000.toml").read_text() == (
        _FAMILY_DIR / "gaussian-cg50-1000.toml"
    ).read_text()
    assert (tmp_path / "gaussian-cg50-5000.toml").read_text() == (
        _FAMILY_DIR / "gaussian-cg50-5000.toml"
    ).read_text()


def test_expand_sweep_is_generic_across_an_unknown_strategy_and_extra_source_fields(
    dataset_sweep_module: ModuleType, tmp_path: Path
) -> None:
    """Proves the renderer has no hardcoded strategy/field knowledge (OCP fix).

    Uses a made-up strategy name, a swept field the script has never seen
    (``every_n`` instead of ``samples``/``cg_iters``), and an extra
    ``[source]`` field (``rhs_path``) the script has no special handling
    for — all must pass straight through.
    """
    spec = {
        "id": "made-up-dataset",
        "source": {
            "matrix_path": "${NEURALLS_RAW_DIR}/matrices/made-up.txt",
            "rhs_path": "${NEURALLS_RAW_DIR}/rhs/made-up.txt",
        },
        "generation": {
            "normalize": "rhs",
            "shuffle": False,
            "seed": 7,
            "strategy": [{"name": "totally_new_strategy", "every_n": [5, 10]}],
        },
        "output": {"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    }
    written = dataset_sweep_module.expand_sweep(spec, output_dir=tmp_path, filename_stem="made-up")
    first = (tmp_path / "made-up-5.toml").read_text()
    assert 'id = "made-up-dataset-5"' in first
    assert 'rhs_path = "${NEURALLS_RAW_DIR}/rhs/made-up.txt"' in first
    assert 'name = "totally_new_strategy"' in first
    assert "every_n = 5" in first
    assert "shuffle = false" in first
    assert len(written) == 2
