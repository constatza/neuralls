"""Shared pytest fixtures for workflow tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.infrastructure.config.factories import load_job
from dlkit.infrastructure.config.job_config import TrainingJobConfig

from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)

_MINIMAL_TRAINING_JOB_TOML = """
[run]
type = "train"

[model]
name = "ScaleEquivariantEmbeddedFactorizedFFNN"
num_layers = 1
activation = "gelu"

[data]
name = "FlexibleDataset"
batch_size = 4

[data.module]
name = "ArrayDataModule"
module_path = "dlkit.engine.adapters.lightning.datamodules"

[training]
"""

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

EXP_ID_ALPHA: str = "alpha-experiment"
EXP_ID_BETA: str = "beta-experiment"


@pytest.fixture
def minimal_training_job(tmp_path: Path) -> TrainingJobConfig:
    """Self-contained training job config, independent of any repo config file."""
    job_path = tmp_path / "minimal-job.toml"
    job_path.write_text(_MINIMAL_TRAINING_JOB_TOML, encoding="utf-8")
    job = load_job(job_path)
    assert isinstance(job, TrainingJobConfig)
    return job


@pytest.fixture
def checkpoint_alpha() -> Path:
    """Checkpoint path for the alpha experiment (never opened)."""
    return Path("alpha/model.ckpt")


@pytest.fixture
def checkpoint_beta() -> Path:
    """Checkpoint path for the beta experiment (never opened)."""
    return Path("beta/model.ckpt")


@pytest.fixture
def jacobi_spec() -> StandardPreconditionerConfig:
    """Non-neural Jacobi preconditioner spec.

    Returns:
        StandardPreconditionerConfig for jacobi.
    """
    return StandardPreconditionerConfig(type=PreconditionerType.JACOBI)


@pytest.fixture
def neural_spec_with_checkpoint(checkpoint_alpha: Path) -> NeuralPreconditionerConfig:
    """Neural preconditioner spec with an explicit checkpoint_path.

    Args:
        checkpoint_alpha: Concrete checkpoint path.

    Returns:
        NeuralPreconditionerConfig with checkpoint_path set.
    """
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        checkpoint_path=checkpoint_alpha,
    )


@pytest.fixture
def neural_spec_with_experiment() -> NeuralPreconditionerConfig:
    """Neural preconditioner spec with an assignment reference (no checkpoint).

    Returns:
        NeuralPreconditionerConfig with assignment set but checkpoint_path=None.
    """
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        assignment=EXP_ID_ALPHA,
    )


@pytest.fixture
def neural_spec_invalid() -> NeuralPreconditionerConfig:
    """Neural preconditioner spec missing both checkpoint_path and assignment.

    Returns:
        NeuralPreconditionerConfig that should fail validation.
    """
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
    )
