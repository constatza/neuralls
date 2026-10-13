"""Schema validation tests for NeuralPODCoarseningConfig.

Only schema/validity is tested here (per project convention: no tests
asserting on specific values inside a checked-in TOML) — parsing, discriminated
union resolution, and the shared rank validator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    CoarseningConfig,
    NeuralPODCoarseningConfig,
)


def test_neural_pod_coarsening_config_parses_with_checkpoint_path(tmp_path: Path) -> None:
    """A minimal neural_pod coarsening config validates with a direct checkpoint."""
    cfg = NeuralPODCoarseningConfig(
        dataset_dir=tmp_path / "params-dataset",
        checkpoint_path=tmp_path / "model.ckpt",
        input_names=("young_modulus",),
        rank=4,
    )
    assert cfg.method == "neural_pod"
    assert cfg.rank == 4
    assert cfg.input_names == ("young_modulus",)


def test_neural_pod_coarsening_config_requires_input_names(tmp_path: Path) -> None:
    """input_names is required — there is no default that could silently mismatch.

    Unlike the dataset's on-disk params artifact key (an enumeration
    artifact of how the dataset was written), the checkpoint's expected
    keyword-argument names are declared training-side and must be stated
    explicitly here; there is no way to infer them.
    """
    with pytest.raises(ValidationError, match="input_names"):
        NeuralPODCoarseningConfig.model_validate(
            {
                "dataset_dir": str(tmp_path / "params-dataset"),
                "checkpoint_path": str(tmp_path / "model.ckpt"),
            }
        )


def test_neural_pod_coarsening_config_coerces_input_names_list_to_tuple(tmp_path: Path) -> None:
    """A TOML-style list of input names is coerced to a tuple."""
    cfg = NeuralPODCoarseningConfig(
        dataset_dir=tmp_path / "params-dataset",
        checkpoint_path=tmp_path / "model.ckpt",
        input_names=["young_modulus", "poisson_ratio"],
    )
    assert cfg.input_names == ("young_modulus", "poisson_ratio")


def test_amg_config_resolves_neural_pod_coarsening_via_discriminator(tmp_path: Path) -> None:
    """AMGPreconditionerConfig.coarsening resolves to NeuralPODCoarseningConfig by method."""
    config = AMGPreconditionerConfig(
        coarsening={
            "method": "neural_pod",
            "dataset_dir": str(tmp_path / "params-dataset"),
            "checkpoint_path": str(tmp_path / "model.ckpt"),
            "input_names": ["young_modulus"],
            "rank": 0.99,
        },
    )
    assert isinstance(config.coarsening, NeuralPODCoarseningConfig)
    assert config.coarsening.rank == 0.99


def test_neural_pod_coarsening_config_rejects_invalid_energy_rank(tmp_path: Path) -> None:
    """Float rank outside (0, 1] is rejected, matching PODCoarseningConfig's rule."""
    with pytest.raises(ValidationError, match="energy threshold"):
        NeuralPODCoarseningConfig(
            dataset_dir=tmp_path / "params-dataset",
            checkpoint_path=tmp_path / "model.ckpt",
            input_names=("young_modulus",),
            rank=1.5,
        )


def test_coarsening_config_union_round_trips_neural_pod(tmp_path: Path) -> None:
    """The CoarseningConfig discriminated union round-trips a neural_pod entry."""
    adapter = TypeAdapter(CoarseningConfig)
    parsed = adapter.validate_python(
        {
            "method": "neural_pod",
            "dataset_dir": str(tmp_path / "params-dataset"),
            "checkpoint_path": str(tmp_path / "model.ckpt"),
            "input_names": ["young_modulus"],
        }
    )
    assert isinstance(parsed, NeuralPODCoarseningConfig)
