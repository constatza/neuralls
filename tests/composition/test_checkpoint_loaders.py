"""Checkpoint loaders carry the one device composition decided, and nothing builds a default."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from neuralls.composition.preconditioners import checkpoint_store
from neuralls.composition.preconditioners.checkpoint_store import (
    CheckpointLoaders,
    MissingCheckpointLoadersError,
)
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter

_DEVICE = torch.device("meta")
_CHECKPOINT = Path("virtual/model.ckpt")


def test_dlkit_loaders_open_inference_predictors_on_their_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = MagicMock()
    monkeypatch.setattr(checkpoint_store, "create_inference_predictor", opened)

    loaders = CheckpointLoaders.dlkit(_DEVICE)
    loaders.inference_predictor_factory(_CHECKPOINT)

    assert (loaders.device, type(loaders.adapter), opened.call_args) == (
        _DEVICE,
        DLKitAdapter,
        ((_CHECKPOINT,), {"device": _DEVICE}),
    )


def test_a_neural_preconditioner_without_loaders_is_rejected() -> None:
    config = NeuralPreconditionerConfig(type=PreconditionerType.NEURAL, checkpoint_path=_CHECKPOINT)

    with pytest.raises(MissingCheckpointLoadersError):
        create_preconditioner(torch.eye(3, dtype=torch.float64), config)
