"""Load a preconditioner config's checkpoints ahead of its timed build.

Which checkpoints a config depends on comes from the `CheckpointRefBearing`
protocol (the one traversal every consumer shares); how each kind of
checkpoint is loaded is a small mapping from the ref's type to a preloader, so
a new checkpoint-backed kind needs one entry here and no edit to the store or
to the runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from torchalg.preconditioners.ports import PredictorAdapter

from neuralls.composition.preconditioners.checkpoint_store import (
    CheckpointStore,
    PodBasisKey,
    PredictorFactory,
    PredictorKey,
)
from neuralls.composition.preconditioners.coarsening import load_fitted_pod_model
from neuralls.platform.config.models.preconditioner import (
    CheckpointRefBearing,
    NeuralCheckpointRef,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
)
from neuralls.platform.dlkit.inference_adapter import create_inference_predictor
from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter


@dataclass(frozen=True, slots=True)
class CheckpointLoaders:
    """The injected collaborators that turn a checkpoint path into a loaded object."""

    adapter: PredictorAdapter
    inference_predictor_factory: PredictorFactory

    @classmethod
    def default(cls, adapter: PredictorAdapter | None = None) -> CheckpointLoaders:
        """DLKit-backed loaders, with ``adapter`` replacing the DLKit adapter when given."""
        return cls(adapter or DLKitAdapter(), create_inference_predictor)


type RefPreloader = Callable[
    [NeuralCheckpointRef, Path, torch.Tensor, CheckpointStore, CheckpointLoaders], None
]


def _preload_pod_basis(
    ref: NeuralCheckpointRef,
    path: Path,
    matrix: torch.Tensor,
    store: CheckpointStore,
    loaders: CheckpointLoaders,
) -> None:
    del ref, loaders
    key = PodBasisKey(path, matrix.dtype, matrix.device)
    store.load_pod_basis(
        key, lambda: load_fitted_pod_model(path).to(dtype=matrix.dtype, device=matrix.device)
    )


def _preload_predictor(
    ref: NeuralCheckpointRef,
    path: Path,
    matrix: torch.Tensor,
    store: CheckpointStore,
    loaders: CheckpointLoaders,
) -> None:
    del matrix
    store.load_predictor(
        PredictorKey(path),
        lambda: loaders.adapter.create_predictor(path, ref.config_path, ref.data_config_path),
    )


def _preload_inference_predictor(
    ref: NeuralCheckpointRef,
    path: Path,
    matrix: torch.Tensor,
    store: CheckpointStore,
    loaders: CheckpointLoaders,
) -> None:
    del ref, matrix
    store.load_inference_predictor(
        PredictorKey(path), lambda: loaders.inference_predictor_factory(path)
    )


_REF_PRELOADERS: Mapping[type[NeuralCheckpointRef], RefPreloader] = MappingProxyType(
    {
        NeuralPreconditionerConfig: _preload_predictor,
        NeuralTransferConfig: _preload_predictor,
        PODCoarseningConfig: _preload_pod_basis,
        NeuralPODCoarseningConfig: _preload_inference_predictor,
    }
)
"""How each checkpoint-ref kind is loaded; a kind missing here fails loudly in `preload_checkpoints`."""


def preload_checkpoints(
    config: PreconditionerConfig,
    matrix: torch.Tensor,
    store: CheckpointStore,
    loaders: CheckpointLoaders,
) -> None:
    """Load every checkpoint ``config`` depends on into ``store`` (a no-op for other configs).

    Idempotent per checkpoint, so a later matrix of the same dataset (or a timing
    repeat) finds everything already loaded.

    Raises:
        TypeError: If the config exposes a checkpoint ref of a kind with no preloader.
    """
    if not isinstance(config, CheckpointRefBearing):
        return
    for _, ref in config.checkpoint_refs():
        path = ref.active_checkpoint_path
        if path is None:
            continue
        preloader = _REF_PRELOADERS.get(type(ref))
        if preloader is None:
            raise TypeError(f"No checkpoint preloader for {type(ref).__name__}")
        preloader(ref, path, matrix, store, loaders)
