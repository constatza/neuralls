"""Tests for structured linear model configs and registry layouts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from neuralls.composition.assignments.assembler import (
    AssignmentIdentity,
    load_assignment,
    load_validated_case_config,
)
from neuralls.platform.config.dlkit_bridge import load_job_config
from neuralls.platform.config.registry import list_assignment_bindings

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None,
    reason="dlkit is not importable in this environment",
)


def _write_job_config(path: Path, model_name: str, checkpoint_name: str) -> Path:
    """Write one isolated thin job config and its model profile."""
    model_profile_path = path.with_name(f"{path.stem}-profile.toml")
    model_profile_path.write_text(
        f"""
[model]
name = "{model_name}"

[data]
name = "FlexibleDataset"

[data.module]
name = "ArrayDataModule"
"""
    )
    path.write_text(
        f"""
[run]
type = "train"
seed = 42
precision = "64"
model = "{model_profile_path.name}"
data = "{model_profile_path.name}"

[training.trainer]
max_epochs = 1
accelerator = "cpu"
enable_checkpointing = true

[[training.trainer.callbacks]]
name = "ModelCheckpoint"
filename = "{checkpoint_name}"
monitor = "val/loss"

[training.optimizer]
default_optimizer = {{ name = "AdamW", lr = 1e-3 }}
"""
    )
    return path


def _write_master_registry(
    path: Path,
    *,
    datasets: list[tuple[str, str]],
    jobs: list[tuple[str, str]],
    experiments: list[tuple[str, str, str]],
) -> Path:
    """Write a minimal case config with explicit tables."""
    lines = [
        "[mlflow]",
        'tracking_uri = "http://127.0.0.1:5000"',
        "",
        "[names]",
        'training = "Train"',
        'comparison = "Comparisons"',
        "",
    ]

    for dataset_id, dataset_path in datasets:
        lines.extend(
            [
                "[[datasets]]",
                f'id = "{dataset_id}"',
                f'path = "{Path(dataset_path).as_posix()}"',
                "",
            ]
        )

    for job_id, job_path in jobs:
        lines.extend(
            [
                "[[jobs]]",
                f'id = "{job_id}"',
                f'path = "{Path(job_path).as_posix()}"',
                "",
            ]
        )

    for experiment_id, dataset_id, job_id in experiments:
        lines.extend(
            [
                "[[assignments]]",
                f'id = "{experiment_id}"',
                f'dataset = "{dataset_id}"',
                f'job = "{job_id}"',
                "",
            ]
        )

    path.write_text("\n".join(lines))
    return path


@pytest.mark.parametrize(
    ("config_name", "model_name"),
    [
        ("symmetric.toml", "NormScaledSymmetricLinear"),
        ("spd.toml", "NormScaledSPDLinear"),
        ("factorized.toml", "NormScaledFactorizedLinear"),
    ],
)
def test_structured_linear_model_configs_load(
    tmp_path: Path,
    config_name: str,
    model_name: str,
    neuralls_settings,
) -> None:
    """Structured linear model configs parse through dlkit."""
    model_path = _write_job_config(
        tmp_path / config_name,
        model_name=model_name,
        checkpoint_name=config_name.removesuffix(".toml"),
    )

    settings = load_job_config(model_path, neuralls_settings)
    assert settings.model is not None
    assert settings.model.name == model_name


def test_legacy_model_workflow_toml_is_rejected(tmp_path: Path, neuralls_settings) -> None:
    """Legacy uppercase manifests fail fast via DLKit's own [run] validation."""
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text(
        '[SESSION]\nseed = 42\nworkflow = "train"\n\n[MODEL]\nname = "NormScaledSymmetricLinear"\n\n[TRAINING]\n[TRAINING.trainer]\nmax_epochs = 1\n\n[OPTIMIZATION]\n\n[[OPTIMIZATION.stages]]\n[OPTIMIZATION.stages.optimizer]\nname = "AdamW"\nlr = 1e-3'
    )

    with pytest.raises(Exception, match="run.type"):
        load_job_config(bad_config, neuralls_settings)


def test_default_structured_linear_registry_has_expected_models_and_experiments(
    tmp_path: Path,
) -> None:
    """A structured-linear registry resolves the intended 3 x 3 experiment matrix."""
    config_path = _write_master_registry(
        tmp_path / "experiments.toml",
        datasets=[
            ("residuals-100", "datasets/residuals-100.toml"),
            ("eig-rhs-smallest", "datasets/eig-rhs-smallest.toml"),
            ("eig-solutions-smallest", "datasets/eig-solutions-smallest.toml"),
        ],
        jobs=[
            ("symmetric", "models/symmetric.toml"),
            ("spd", "models/spd.toml"),
            ("factorized", "models/factorized.toml"),
        ],
        experiments=[
            ("eig-rhs-smallest-symmetric", "eig-rhs-smallest", "symmetric"),
            ("eig-rhs-smallest-spd", "eig-rhs-smallest", "spd"),
            ("eig-rhs-smallest-factorized", "eig-rhs-smallest", "factorized"),
            ("eig-solutions-smallest-symmetric", "eig-solutions-smallest", "symmetric"),
            ("eig-solutions-smallest-spd", "eig-solutions-smallest", "spd"),
            ("eig-solutions-smallest-factorized", "eig-solutions-smallest", "factorized"),
            ("residuals-symmetric", "residuals-100", "symmetric"),
            ("residuals-spd", "residuals-100", "spd"),
            ("residuals-factorized", "residuals-100", "factorized"),
        ],
    )

    cfg, config_dir = load_validated_case_config(config_path)
    bindings = list_assignment_bindings(cfg, config_dir)

    assert [entry.id for entry in cfg.jobs] == ["symmetric", "spd", "factorized"]
    assert [entry.id for entry in cfg.datasets] == [
        "residuals-100",
        "eig-rhs-smallest",
        "eig-solutions-smallest",
    ]
    assert [binding.assignment_id for binding in bindings] == [
        "eig-rhs-smallest-symmetric",
        "eig-rhs-smallest-spd",
        "eig-rhs-smallest-factorized",
        "eig-solutions-smallest-symmetric",
        "eig-solutions-smallest-spd",
        "eig-solutions-smallest-factorized",
        "residuals-symmetric",
        "residuals-spd",
        "residuals-factorized",
    ]


def test_linear_registry_uses_single_normalized_model(tmp_path: Path) -> None:
    """A linear-focused registry can use one normalized-loss model across cases."""
    config_path = _write_master_registry(
        tmp_path / "case-linear.toml",
        datasets=[
            ("spectral", "datasets/solutions.toml"),
            ("residuals-100", "datasets/residuals-100.toml"),
            ("eig-rhs-largest", "datasets/eig-rhs-largest.toml"),
            ("eig-rhs-smallest", "datasets/eig-rhs-smallest.toml"),
            ("eig-solutions-largest", "datasets/eig-solutions-largest.toml"),
            ("eig-solutions-smallest", "datasets/eig-solutions-smallest.toml"),
        ],
        jobs=[
            ("linear", "models/linear.toml"),
        ],
        experiments=[
            ("spectral", "spectral", "linear"),
            ("eig-rhs-largest", "eig-rhs-largest", "linear"),
            ("eig-rhs-smallest", "eig-rhs-smallest", "linear"),
            ("eig-solutions-largest", "eig-solutions-largest", "linear"),
            ("eig-solutions-smallest", "eig-solutions-smallest", "linear"),
            ("residuals", "residuals-100", "linear"),
        ],
    )

    cfg, config_dir = load_validated_case_config(config_path)
    bindings = list_assignment_bindings(cfg, config_dir)

    assert [entry.id for entry in cfg.jobs] == ["linear"]
    assert [binding.assignment_id for binding in bindings] == [
        "spectral",
        "eig-rhs-largest",
        "eig-rhs-smallest",
        "eig-solutions-largest",
        "eig-solutions-smallest",
        "residuals",
    ]


def test_load_experiment_supports_structured_linear_job(tmp_path: Path) -> None:
    """A structured job config loads through the training entry point."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    data_config_path = tmp_path / "data.toml"
    data_config_path.write_text('id = "dummy_dataset"\n[source]\nmatrix_path = "matrix.txt"\n')
    job_config_path = _write_job_config(
        tmp_path / "symmetric.toml",
        model_name="NormScaledSymmetricLinear",
        checkpoint_name="symmetric",
    )

    experiment = load_assignment(
        job_config_path=job_config_path,
        data_config_path=data_config_path,
        output_root=output_root,
        identity=AssignmentIdentity(dataset_registry_id="dummy-dataset"),
    )

    assert experiment.spec.job_config_path.name == "symmetric.toml"
    assert experiment.spec.dataset_id == "dummy-dataset"
    assert experiment.settings.model.name == "NormScaledSymmetricLinear"
