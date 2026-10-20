"""Loaded checkpoints shared across builds: typed keys, the read/write split, and the adapters.

A checkpoint is loaded once per dataset, *before* a preconditioner's timed
build, and every build then borrows the loaded object. Builders see only the
read side (`PreloadedCheckpoints`); `CheckpointStore` owns the objects and is the
only place that releases them. torchalg's `NeuralPreconditioner` and neural
coarsening never release a predictor they were given, so borrowing is safe; only
a consumer written as ``with factory(path) as predictor`` (neural POD-2G's
inference predictor) releases on exit, and gets a non-owning view.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import torch
from loguru import logger
from torchalg.preconditioners.ports import ExtraInputPredictorPort

from neuralls.domain.inference_ports import InferencePredictorPort
from neuralls.platform.dlkit.inference_adapter import NonOwningInferencePredictor

type PredictorFactory = Callable[[Path], InferencePredictorPort]
"""Opens an inference predictor for a checkpoint; the default is `create_inference_predictor`."""


@dataclass(frozen=True, slots=True)
class PodBasisKey:
    """A fitted POD basis module, already moved to the matrix's dtype and device."""

    path: Path
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, slots=True)
class PredictorKey:
    """A predictor (adapter-style or batch-inference) loaded from one checkpoint."""

    path: Path


class PreloadedCheckpoints(Protocol):
    """Read side: what a builder may ask for before falling back to loading itself."""

    def pod_basis(self, key: PodBasisKey) -> torch.nn.Module | None:
        """The preloaded fitted POD module, or ``None`` when it was not preloaded."""
        ...

    def predictor(self, key: PredictorKey) -> ExtraInputPredictorPort | None:
        """The preloaded predictor (owned by the store; borrow it, never release it), or ``None``."""
        ...

    def inference_predictor(self, key: PredictorKey) -> InferencePredictorPort | None:
        """A non-owning view of the preloaded batch-inference predictor, or ``None``."""
        ...


class NoPreloadedCheckpoints:
    """Null object: nothing is preloaded, so every builder loads inside its own build."""

    def pod_basis(self, key: PodBasisKey) -> torch.nn.Module | None:
        del key
        return None

    def predictor(self, key: PredictorKey) -> ExtraInputPredictorPort | None:
        del key
        return None

    def inference_predictor(self, key: PredictorKey) -> InferencePredictorPort | None:
        del key
        return None


NO_PRELOADED = NoPreloadedCheckpoints()


class CheckpointStore:
    """Write side: owns every loaded checkpoint object until `release`.

    Loaders are injected callables, so this class knows nothing about torch
    checkpoints or DLKit. Each ``load_*`` is idempotent: a key already loaded is
    never loaded again, which is what makes the second matrix of a method (and
    every timing repeat) free.
    """

    def __init__(self) -> None:
        self._pod_bases: dict[PodBasisKey, torch.nn.Module] = {}
        self._predictors: dict[PredictorKey, ExtraInputPredictorPort] = {}
        self._inference_predictors: dict[PredictorKey, InferencePredictorPort] = {}

    def __enter__(self) -> Self:
        """Enter the store's scope; leaving it releases everything loaded."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release every loaded checkpoint."""
        del exc_type, exc, tb
        self.release()

    def load_pod_basis(
        self, key: PodBasisKey, loader: Callable[[], torch.nn.Module]
    ) -> torch.nn.Module:
        """Load (once) the POD module for ``key``."""
        if key not in self._pod_bases:
            self._pod_bases[key] = self._timed(key.path, loader)
        return self._pod_bases[key]

    def load_predictor(
        self, key: PredictorKey, loader: Callable[[], ExtraInputPredictorPort]
    ) -> ExtraInputPredictorPort:
        """Load (once) the adapter-style predictor for ``key``."""
        if key not in self._predictors:
            self._predictors[key] = self._timed(key.path, loader)
        return self._predictors[key]

    def load_inference_predictor(
        self, key: PredictorKey, loader: Callable[[], InferencePredictorPort]
    ) -> InferencePredictorPort:
        """Load (once) the batch-inference predictor for ``key``."""
        if key not in self._inference_predictors:
            self._inference_predictors[key] = self._timed(key.path, loader)
        return self._inference_predictors[key]

    def pod_basis(self, key: PodBasisKey) -> torch.nn.Module | None:
        return self._pod_bases.get(key)

    def predictor(self, key: PredictorKey) -> ExtraInputPredictorPort | None:
        return self._predictors.get(key)

    def inference_predictor(self, key: PredictorKey) -> InferencePredictorPort | None:
        owner = self._inference_predictors.get(key)
        return NonOwningInferencePredictor(owner) if owner is not None else None

    def release(self) -> None:
        """Release every owned predictor exactly once and drop all loaded objects."""
        for predictor in (*self._predictors.values(), *self._inference_predictors.values()):
            predictor.cleanup()
        self._pod_bases.clear()
        self._predictors.clear()
        self._inference_predictors.clear()

    def _timed[T](self, path: Path, loader: Callable[[], T]) -> T:
        loaded = loader()
        logger.info("Preloaded checkpoint '{}'.", path)
        return loaded


def preloaded_inference_factory(
    base: PredictorFactory, preloaded: PreloadedCheckpoints
) -> PredictorFactory:
    """A predictor factory that serves preloaded inference predictors, else calls ``base``."""

    def factory(path: Path) -> InferencePredictorPort:
        preloaded_predictor = preloaded.inference_predictor(PredictorKey(path))
        return preloaded_predictor if preloaded_predictor is not None else base(path)

    return factory
