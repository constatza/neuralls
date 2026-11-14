"""Load a preconditioner config's checkpoints ahead of its timed build.

Which checkpoints a config depends on comes from the `CheckpointRefBearing`
protocol (the one traversal every consumer shares); how each kind of
checkpoint is loaded is a small mapping from the ref's type to a preloader, so
a new checkpoint-backed kind needs one entry here and no edit to the store or
to the runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

import torch

from neuralls.composition.preconditioners.checkpoint_store import (
    CheckpointLoaders,
    CheckpointStore,
    PodBasisKey,
    PredictorKey,
    require_loaders,
)
from neuralls.composition.preconditioners.coarsening import load_fitted_pod_model
from neuralls.platform.config.models.checkpoint_refs import (
    CheckpointRefBearing,
    NeuralCheckpointRef,
)
from neuralls.platform.config.models.coarsening import (
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PreconditionerConfig,
)

type RefPreloader = Callable[
    [NeuralCheckpointRef, Path, torch.Tensor, CheckpointStore, CheckpointLoaders | None], None
]


def _preload_pod_basis(
    ref: NeuralCheckpointRef,
    path: Path,
    matrix: torch.Tensor,
    store: CheckpointStore,
    loaders: CheckpointLoaders | None,
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
    loaders: CheckpointLoaders | None,
) -> None:
    del matrix
    store.load_predictor(
        PredictorKey(path),
        lambda: require_loaders(loaders, f"preloading {path}").adapter.create_predictor(
            path, ref.config_path, ref.data_config_path
        ),
    )


def _preload_inference_predictor(
    ref: NeuralCheckpointRef,
    path: Path,
    matrix: torch.Tensor,
    store: CheckpointStore,
    loaders: CheckpointLoaders | None,
) -> None:
    del ref, matrix
    store.load_inference_predictor(
        PredictorKey(path),
        lambda: require_loaders(loaders, f"preloading {path}").inference_predictor_factory(path),
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
    loaders: CheckpointLoaders | None,
) -> None:
    """Load every checkpoint ``config`` depends on into ``store`` (a no-op for other configs).

    Idempotent per checkpoint, so a later matrix of the same dataset (or a timing
    repeat) finds everything already loaded. ``loaders`` is needed only to load a predictor; a
    config without one, or a fitted POD-2G basis, loads without them.

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
