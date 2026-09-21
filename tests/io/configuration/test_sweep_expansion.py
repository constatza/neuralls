"""Tests for on-demand sweep expansion during case-config loading."""

from __future__ import annotations

from pathlib import Path

from neuralls.platform.config.loaders import load_case_config
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.config.sweep_expansion import ensure_generated_dataset_config


def test_missing_generated_dataset_is_created_on_load(
    sweep_case_config: Path, generated_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """A referenced `_generated/` file is expanded from its sweep and fills the dataset id."""
    cfg = load_case_config(sweep_case_config, neuralls_settings)

    assert generated_path.exists()
    assert cfg.datasets[0].id == "sweep-ds-10"


def test_existing_generated_dataset_is_not_rewritten_without_force(
    sweep_case_config: Path, generated_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """Without force an existing generated file is left untouched."""
    load_case_config(sweep_case_config, neuralls_settings)
    generated_path.write_text(generated_path.read_text() + "# edited\n", encoding="utf-8")

    load_case_config(sweep_case_config, neuralls_settings)

    assert generated_path.read_text().endswith("# edited\n")


def test_force_expand_recreates_generated_dataset(
    sweep_case_config: Path, generated_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """`force_expand` overwrites an existing generated file from its sweep."""
    load_case_config(sweep_case_config, neuralls_settings)
    generated_path.write_text(generated_path.read_text() + "# edited\n", encoding="utf-8")

    load_case_config(sweep_case_config, neuralls_settings, force_expand=True)

    assert "# edited" not in generated_path.read_text()


def test_generated_path_without_sweep_source_is_left_missing(tmp_path: Path) -> None:
    """A `_generated/` path no sweep produces is not created; the caller's error stands."""
    missing = tmp_path / "datasets" / "_generated" / "nope.toml"
    (tmp_path / "datasets").mkdir()

    ensure_generated_dataset_config(missing)

    assert not missing.exists()


def test_path_outside_generated_dir_is_never_expanded(
    sweep_dataset_dir: Path,
) -> None:
    """Only `_generated/` paths trigger expansion."""
    typo = sweep_dataset_dir / "gaussian-cg-10.toml"

    ensure_generated_dataset_config(typo, force=True)

    assert not (sweep_dataset_dir / "_generated").exists()
