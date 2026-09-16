"""Tests for the pure helpers in ``scripts/expand_dataset_sweep.py``.

The golden-file tests double as the correctness proof that the generic
renderer reproduces real ``*.sweep.toml`` expansions exactly — not just
something similar. Expansions themselves are gitignored build artifacts (see
``_GENERATED_SUBDIR_NAME`` in the script under test) not guaranteed to exist
on a fresh checkout, so these tests regenerate their own golden copies from
the real, committed ``configs/datasets/train/rectangular-high-condition/
*.sweep.toml`` sources into an isolated ``tmp_path`` rather than reading
``_generated/`` directly — see ``rectangular_high_condition_golden_dir``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAMILY_DIR = _REPO_ROOT / "configs" / "datasets" / "train" / "rectangular-high-condition"


@pytest.fixture(scope="session")
def rectangular_high_condition_golden_dir(
    dataset_sweep_module: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Fresh expansions of the real, committed rectangular-high-condition sweep specs.

    Args:
        dataset_sweep_module: The loaded ``expand_dataset_sweep`` module.
        tmp_path_factory: Session-scoped temp directory factory.

    Returns:
        A directory containing every dataset config expanded from
        ``gaussian-cg0.sweep.toml`` and ``gaussian-cg50.sweep.toml``.
    """
    out_dir = tmp_path_factory.mktemp("rectangular-high-condition-golden")
    for sweep_name in ("gaussian-cg0.sweep.toml", "gaussian-cg50.sweep.toml"):
        spec = dataset_sweep_module._load_sweep_spec(_FAMILY_DIR / sweep_name)
        filename_stem = sweep_name[: -len(dataset_sweep_module._SWEEP_SUFFIX)]
        dataset_sweep_module.expand_sweep(spec, output_dir=out_dir, filename_stem=filename_stem)
    return out_dir


def test_render_dataset_toml_matches_committed_gaussian_residuals_file(
    rectangular_high_condition_golden_dir: Path,
    dataset_sweep_module: ModuleType,
) -> None:
    """A gaussian_residuals-shaped render matches the real gaussian-cg50-1000.toml exactly."""
    content = dataset_sweep_module._render_dataset_toml(
        id_="gaussian-cg50-rectangular-high-condition-1000",
        source={"matrix_path": "${NEURALLS_RAW_DIR}/matrices/rectangular-high-condition.txt"},
        generation_header={"normalize": "matrix", "shuffle": True, "seed": 42},
        strategy={"name": "gaussian_residuals", "samples": 1000, "stop": 50},
        strategy_comment=None,
        output={"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    )
    assert (
        content == (rectangular_high_condition_golden_dir / "gaussian-cg50-1000.toml").read_text()
    )


def test_render_dataset_toml_matches_committed_gaussian_forward_file(
    rectangular_high_condition_golden_dir: Path,
    dataset_sweep_module: ModuleType,
) -> None:
    """A gaussian_forward-shaped render with strategy_comment matches gaussian-cg0-1000.toml exactly."""
    comment = [
        '"0-CG": the `stop` field on the residuals/trace strategies has a hard',
        "floor of 1 (and even stop=1 emits 2 rows per system - iteration 0 AND",
        "iteration 1, not just the untraced solution). gaussian_forward is the true",
        "0-CG case - x ~ N(mu, sigma), b = A @ x, no CG solve at all.",
        "POD-2G fitting never sees real archived solutions - those are comparison-only.",
    ]
    content = dataset_sweep_module._render_dataset_toml(
        id_="gaussian-cg0-rectangular-high-condition-1000",
        source={"matrix_path": "${NEURALLS_RAW_DIR}/matrices/rectangular-high-condition.txt"},
        generation_header={"normalize": "matrix", "shuffle": True, "seed": 42},
        strategy={"name": "gaussian_forward", "samples": 1000},
        strategy_comment=comment,
        output={"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    )
    assert content == (rectangular_high_condition_golden_dir / "gaussian-cg0-1000.toml").read_text()


def test_find_swept_field_rejects_no_list_valued_field(dataset_sweep_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="no list-valued field"):
        dataset_sweep_module._find_swept_field({"name": "gaussian_residuals", "samples": 500})


def test_find_swept_field_rejects_multiple_list_valued_fields(
    dataset_sweep_module: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="only one field may be swept"):
        dataset_sweep_module._find_swept_field(
            {"name": "gaussian_residuals", "samples": [1, 2], "stop": [10, 20]}
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
    dataset_sweep_module: ModuleType,
    rectangular_high_condition_golden_dir: Path,
    tmp_path: Path,
) -> None:
    spec = {
        "id": "gaussian-cg50-rectangular-high-condition",
        "source": {"matrix_path": "${NEURALLS_RAW_DIR}/matrices/rectangular-high-condition.txt"},
        "generation": {
            "normalize": "matrix",
            "shuffle": True,
            "seed": 42,
            "strategy": [{"name": "gaussian_residuals", "samples": [1000, 5000], "stop": 50}],
        },
        "output": {"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    }
    written = dataset_sweep_module.expand_sweep(
        spec, output_dir=tmp_path, filename_stem="gaussian-cg50"
    )
    assert [p.name for p in written] == ["gaussian-cg50-1000.toml", "gaussian-cg50-5000.toml"]
    assert (tmp_path / "gaussian-cg50-1000.toml").read_text() == (
        rectangular_high_condition_golden_dir / "gaussian-cg50-1000.toml"
    ).read_text()
    assert (tmp_path / "gaussian-cg50-5000.toml").read_text() == (
        rectangular_high_condition_golden_dir / "gaussian-cg50-5000.toml"
    ).read_text()


def test_expand_sweep_id_template_places_value_mid_id(
    dataset_sweep_module: ModuleType, tmp_path: Path
) -> None:
    """`id_template`/`filename_template` put the swept value in the middle of the id.

    Unlike a trailing axis (`samples`), `stop` belongs between the strategy
    name and the family: `gaussian-cg10-45x15`, not `gaussian-cg-45x15-10`.
    """
    spec = {
        "id": "gaussian-cg-45x15",
        "id_template": "gaussian-cg{value}-45x15",
        "filename_template": "gaussian-cg{value}.toml",
        "source": {"matrix_path": "${NEURALLS_RAW_DIR}/matrices/45x15.txt"},
        "generation": {
            "normalize": "matrix",
            "shuffle": True,
            "seed": 42,
            "strategy": [{"name": "gaussian_residuals", "samples": 5000, "stop": [10, 50]}],
        },
        "output": {"data_dir": "${NEURALLS_PROCESSED_DIR}"},
    }
    written = dataset_sweep_module.expand_sweep(spec, output_dir=tmp_path, filename_stem="unused")

    assert [p.name for p in written] == ["gaussian-cg10.toml", "gaussian-cg50.toml"]
    assert 'id = "gaussian-cg10-45x15"' in (tmp_path / "gaussian-cg10.toml").read_text()
    assert 'id = "gaussian-cg50-45x15"' in (tmp_path / "gaussian-cg50.toml").read_text()


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


@pytest.fixture
def sweep_spec_toml_lines() -> list[str]:
    """Minimal valid ``*.sweep.toml`` body, one list-valued strategy field."""
    return [
        'id = "case-a-gaussian"',
        "",
        "[source]",
        'matrix_path = "${NEURALLS_RAW_DIR}/matrices/case-a.txt"',
        "",
        "[generation]",
        'normalize = "matrix"',
        "shuffle = true",
        "seed = 42",
        "",
        "[[generation.strategy]]",
        'name = "gaussian_residuals"',
        "samples = 1000",
        "stop = [10, 50]",
        "",
        "[output]",
        'data_dir = "${NEURALLS_PROCESSED_DIR}"',
    ]


@pytest.fixture
def sweep_tree(tmp_path: Path, sweep_spec_toml_lines: list[str]) -> Path:
    """A small ``configs``-shaped tree with two sweep specs in different subdirectories."""
    case_a_dir = tmp_path / "configs" / "datasets" / "train" / "case-a"
    case_b_dir = tmp_path / "configs" / "datasets" / "train" / "case-b"
    case_a_dir.mkdir(parents=True)
    case_b_dir.mkdir(parents=True)

    (case_a_dir / "gaussian-cg.sweep.toml").write_text(
        "\n".join(sweep_spec_toml_lines) + "\n", encoding="utf-8"
    )
    case_b_lines = [line.replace("case-a", "case-b") for line in sweep_spec_toml_lines]
    (case_b_dir / "gaussian-cg.sweep.toml").write_text(
        "\n".join(case_b_lines) + "\n", encoding="utf-8"
    )
    return tmp_path / "configs"


def test_expand_all_writes_each_sweep_files_outputs_under_its_own_generated_dir(
    dataset_sweep_module: ModuleType, sweep_tree: Path
) -> None:
    """`expand_all` discovers every sweep file under root and expands it in place.

    Each sweep file's outputs land in a `_generated/` subdirectory sibling
    to that sweep file, not in one shared output directory.
    """
    written = dataset_sweep_module.expand_all(sweep_tree)

    written_names = {p.name for p in written}
    assert written_names == {"gaussian-cg-10.toml", "gaussian-cg-50.toml"}
    assert len(written) == 4

    case_a_generated = sweep_tree / "datasets" / "train" / "case-a" / "_generated"
    case_b_generated = sweep_tree / "datasets" / "train" / "case-b" / "_generated"
    assert (case_a_generated / "gaussian-cg-10.toml").exists()
    assert (case_a_generated / "gaussian-cg-50.toml").exists()
    assert (case_b_generated / "gaussian-cg-10.toml").exists()
    assert (case_b_generated / "gaussian-cg-50.toml").exists()
    assert 'id = "case-a-gaussian-10"' in (case_a_generated / "gaussian-cg-10.toml").read_text()
    assert 'id = "case-b-gaussian-50"' in (case_b_generated / "gaussian-cg-50.toml").read_text()


def test_expand_all_honors_custom_generated_subdir_name(
    dataset_sweep_module: ModuleType, sweep_tree: Path
) -> None:
    """`expand_all` writes into a custom subdirectory name when overridden."""
    dataset_sweep_module.expand_all(sweep_tree, generated_subdir_name="_built")

    case_a_built = sweep_tree / "datasets" / "train" / "case-a" / "_built"
    assert (case_a_built / "gaussian-cg-10.toml").exists()
    assert (case_a_built / "gaussian-cg-50.toml").exists()


def test_main_all_flag_expands_every_sweep_file_under_default_configs_root(
    dataset_sweep_module: ModuleType,
    sweep_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main()` with `--all ROOT` walks ROOT and reports every written file."""
    monkeypatch.setattr(sys, "argv", ["expand_dataset_sweep.py", "--all", str(sweep_tree)])
    exit_code = dataset_sweep_module.main()

    assert exit_code == 0
    case_a_generated = sweep_tree / "datasets" / "train" / "case-a" / "_generated"
    case_b_generated = sweep_tree / "datasets" / "train" / "case-b" / "_generated"
    assert (case_a_generated / "gaussian-cg-10.toml").exists()
    assert (case_b_generated / "gaussian-cg-50.toml").exists()


def test_main_rejects_sweep_file_and_all_together(
    dataset_sweep_module: ModuleType,
    sweep_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sweep_file` and `--all` are mutually exclusive."""
    sweep_file = sweep_tree / "datasets" / "train" / "case-a" / "gaussian-cg.sweep.toml"
    monkeypatch.setattr(
        sys, "argv", ["expand_dataset_sweep.py", str(sweep_file), "--all", str(sweep_tree)]
    )
    with pytest.raises(SystemExit):
        dataset_sweep_module.main()
