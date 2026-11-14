"""CPU `CheckpointLoaders` for tests that inject one collaborator and leave the other unused."""

from __future__ import annotations

from pathlib import Path

import torch
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

from neuralls.composition.preconditioners.checkpoint_store import (
    CheckpointLoaders,
    PredictorFactory,
)
from neuralls.domain.inference_ports import InferencePredictorPort

CPU = torch.device("cpu")


class _UnusedAdapter(PredictorAdapter):
    """Fills the adapter slot of a test that never loads an adapter-style predictor."""

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        raise AssertionError("this test must not load an adapter-style predictor")


def _unused_factory(checkpoint_path: Path) -> InferencePredictorPort:
    raise AssertionError("this test must not open an inference predictor")


def cpu_loaders(
    adapter: PredictorAdapter | None = None, factory: PredictorFactory | None = None
) -> CheckpointLoaders:
    """Loaders on the CPU; a slot left out raises if the code under test calls it."""
    return CheckpointLoaders(adapter or _UnusedAdapter(), factory or _unused_factory, CPU)
