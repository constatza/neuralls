"""Unit tests for factory threading of extra_input_names to NeuralPreconditioner.

Tests that the composition factory correctly passes extra_input_names from
NeuralPreconditionerConfig to the NeuralPreconditioner instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchalg.preconditioners.implementations.neural import NeuralPreconditioner
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.platform.config.models.preconditioner import NeuralPreconditionerConfig

# ==============================================================================
# Mock Predictor and Adapter
# ==============================================================================


class MockPredictor(ExtraInputPredictorPort):
    """Lightweight mock predictor for testing."""

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return empty tuple — this mock needs no extra inputs."""
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Mock apply: returns input unchanged."""
        return residual.clone()

    def cleanup(self) -> None:
        """No-op cleanup."""


class MockAdapter(PredictorAdapter):
    """Lightweight mock adapter for testing."""

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create mock predictor, raising if the checkpoint is missing."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return MockPredictor()


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def well_conditioned_matrix() -> torch.Tensor:
    """Well-conditioned 4x4 SPD matrix for testing."""
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float64))


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Create mock checkpoint file."""
    checkpoint = tmp_path / "mock_checkpoint.ckpt"
    checkpoint.write_text("mock checkpoint data", encoding="utf-8")
    return checkpoint


@pytest.fixture
def mock_adapter() -> MockAdapter:
    """Create mock adapter."""
    return MockAdapter()


# ==============================================================================
# Tests for extra_input_names Threading
# ==============================================================================


def test_factory_passes_single_extra_input_name(
    well_conditioned_matrix: torch.Tensor,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Factory passes a single extra_input_name to NeuralPreconditioner."""
    config = NeuralPreconditionerConfig(
        checkpoint_path=mock_checkpoint,
        extra_input_names=("matrix",),
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == ("matrix",)


def test_factory_passes_multiple_extra_input_names(
    well_conditioned_matrix: torch.Tensor,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Factory passes multiple extra_input_names to NeuralPreconditioner."""
    config = NeuralPreconditionerConfig(
        checkpoint_path=mock_checkpoint,
        extra_input_names=("matrix", "coordinates", "bc_mask"),
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == ("matrix", "coordinates", "bc_mask")


def test_factory_empty_extra_input_names_by_default(
    well_conditioned_matrix: torch.Tensor,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Factory uses an empty tuple as the default for extra_input_names."""
    config = NeuralPreconditionerConfig(
        checkpoint_path=mock_checkpoint,
        # extra_input_names not specified - should default to ()
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == ()


def test_factory_preserves_extra_input_names_order(
    well_conditioned_matrix: torch.Tensor,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Factory preserves declaration order of extra_input_names."""
    names = ("first", "second", "third")
    config = NeuralPreconditionerConfig(
        checkpoint_path=mock_checkpoint,
        extra_input_names=names,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == names
    assert isinstance(precond.extra_input_names, tuple)
