"""The one-off fit a method runs inside the comparison, before its per-matrix timed setups.

A POD method's basis depends on its fit dataset, never on the matrix it preconditions, so it is
fitted once per method rather than once per matrix, and its wall time and peak memory are the
method's offline ``fit`` cost (`MEASURED`), not part of any timed setup. The measured region
covers reading the train fold, the energy weight and the SVD, as a training job's recorded wall
time covers its own data loading. Every other method has nothing to fit here; neural POD's
snapshots come from its checkpoint's predictions, so it still fits inside its setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch
from loguru import logger
from torchalg.utils.pod_basis import compute_pod_basis

from neuralls.composition.generation.ports import SplitStore
from neuralls.composition.generation.split_pinning import pinned_train_rows
from neuralls.composition.preconditioners.ports import FitDataLoader
from neuralls.composition.preconditioners.reference_matrix import mean_train_matrix
from neuralls.domain.linalg import captured_energy_fraction
from neuralls.domain.solver.models.cost import Known, NotApplicable, StageCost
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.platform.sparse_io.components import to_components
from neuralls.platform.sparse_io.torch_convert import to_torch_csr
from neuralls.shared.device import track_resource_usage
from neuralls.shared.types import CostProvenance, InnerProductKind

MIN_CAPTURED_ENERGY = 0.95
"""Share of the snapshot energy below which a fitted basis is warned about: the rank is too small."""

FIT_DTYPE = torch.float64
"""Precision of the fit; the SVD of nearly dependent snapshots loses accuracy in float32."""


@dataclass(frozen=True, slots=True)
class NoMethodFit:
    """The method has nothing to fit before its timed setups."""


@dataclass(frozen=True, slots=True)
class FittedPOD:
    """A POD basis fitted once for the method, and what fitting it cost."""

    basis: torch.Tensor
    cost: Known


@dataclass(frozen=True, slots=True)
class MethodFitFailed:
    """The method's fit raised; every matrix then records it as a setup failure."""

    error: Exception
    cost: Known


type MethodFit = NoMethodFit | FittedPOD | MethodFitFailed


def fit_cost(fit: MethodFit) -> StageCost:
    """The offline fit stage ``fit`` measured: none when the method had nothing to fit here."""
    match fit:
        case NoMethodFit():
            return NotApplicable()
        case FittedPOD(cost=cost) | MethodFitFailed(cost=cost):
            return cost
        case unreachable:
            assert_never(unreachable)


def fit_method(
    config: PreconditionerConfig,
    *,
    fit_data: FitDataLoader,
    splits: SplitStore,
    device: torch.device,
) -> MethodFit:
    """Fit what ``config`` needs before its timed setups, measuring it on ``device``.

    Args:
        config: The method's preconditioner configuration.
        fit_data: Where the fit dataset's arrays and matrices are read from.
        splits: Where the fit dataset's split is pinned; only train-fold rows are fitted, since
            comparisons draw held-out systems from the test fold.
        device: Device the fit runs and is measured on.

    Returns:
        `NoMethodFit` for a method without a fit stage here, otherwise the fitted basis or the
        failure, each with its measured cost.
    """
    match config:
        case AMGPreconditionerConfig(coarsening=PODCoarseningConfig() as pod):
            return _fit_pod(pod, fit_data=fit_data, splits=splits, device=device)
        case _:
            return NoMethodFit()


def _fit_pod(
    cfg: PODCoarseningConfig, *, fit_data: FitDataLoader, splits: SplitStore, device: torch.device
) -> FittedPOD | MethodFitFailed:
    with track_resource_usage(device) as usage:
        try:
            result: torch.Tensor | Exception = _pod_basis(
                cfg, fit_data=fit_data, splits=splits, device=device
            )
        except (ValueError, RuntimeError) as exc:  # LinAlgError and OOM are RuntimeErrors
            result = exc
    measured = usage()
    cost = Known(measured.wall_time_seconds, measured.peak_memory_bytes, CostProvenance.MEASURED)
    match result:
        case torch.Tensor():
            return FittedPOD(result, cost)
        case Exception():
            return MethodFitFailed(result, cost)
        case unreachable:
            assert_never(unreachable)


def _pod_basis(
    cfg: PODCoarseningConfig, *, fit_data: FitDataLoader, splits: SplitStore, device: torch.device
) -> torch.Tensor:
    """The train-fold POD basis of ``cfg``, in float64 on ``device``; warns on a low capture."""
    train = list(pinned_train_rows(cfg.dataset_dir, splits))
    _, solutions = fit_data.dense(cfg.dataset_dir)
    rows = solutions[train]
    snapshots = torch.as_tensor(rows[: cfg.inline_fit_rows(len(rows))], dtype=FIT_DTYPE).to(device)
    weight = _inner_product_weight(cfg, fit_data=fit_data, splits=splits, device=device)
    basis = compute_pod_basis(snapshots, cfg.rank, inner_product=weight)
    captured = captured_energy_fraction(snapshots, basis, weight=weight)
    if captured < MIN_CAPTURED_ENERGY:
        logger.warning(
            "POD rank {} on {} captures {:.1%} of the snapshot energy, below {:.0%}; "
            "the coarse space misses much of the snapshots.",
            cfg.rank,
            cfg.dataset_dir,
            captured,
            MIN_CAPTURED_ENERGY,
        )
    return basis


def _inner_product_weight(
    cfg: PODCoarseningConfig, *, fit_data: FitDataLoader, splits: SplitStore, device: torch.device
) -> torch.Tensor | None:
    """``None`` for the Euclidean inner product, else the train-fold mean matrix as torch CSR."""
    match cfg.inner_product:
        case InnerProductKind.EUCLIDEAN:
            return None
        case InnerProductKind.ENERGY:
            mean = mean_train_matrix(cfg.dataset_dir, splits, fit_data)
            return to_torch_csr(to_components(mean)).to(device)
        case unreachable:
            assert_never(unreachable)
