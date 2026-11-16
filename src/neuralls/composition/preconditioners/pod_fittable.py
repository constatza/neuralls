"""dlkit `Fittable` adapter over torchalg's `PODCoarseningStrategy`.

Lives in `composition/preconditioners/` (not `domain/solver/`) per
`domain/solver/README.md`'s boundary rule: all preconditioner algorithms live
in `torchalg`, and `neuralls.domain.solver` production code must not hold
preconditioner algorithm classes. This adapter holds no new math at all — it
inherits `build_transfer`/`compute_pod_basis` from `PODCoarseningStrategy`
unchanged — it exists solely so dlkit's job loader (`[model] module_path`,
resolved via `importlib`) has a concrete, importable class to construct for a
`run.type = "fit"` (POD) assignment, mirroring
`composition/preconditioners/factory.py`'s existing role translating between
neuralls' config layer and `torchalg`'s runtime objects.

`PODCoarseningStrategy.fit(snapshots: torch.Tensor)` takes a raw tensor — its
natural call site is `factory.py`'s inline fit, which already has the
snapshot tensor in hand. dlkit's `FitJobConfig`/`OneShotFitExecutor` pipeline
instead calls `model.fit(dataloader)` (see
`dlkit.engine.training.fittable.Fittable`), passing an iterable of batches,
not a tensor — verified directly against the installed `PODCoarseningStrategy`:
`PODCoarseningStrategy(rank=2).fit(a_real_dataloader)` raises
`AttributeError: 'DataLoader' object has no attribute 'shape'`.

This adapter bridges that gap the same way dlkit's own
`TransformChain.fit_from_dataloader()` bridges `IFittableTransformer
.fit(dataloader)` to each wrapped `Transform.fit(data: Tensor)`
(`dlkit.domain.transforms.chain`) — materialize the target array across the
dataloader's batches, then delegate to the tensor-based `fit()`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
from torchalg.utils.device import resolve_device

from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)

_DEFAULT_TARGET_NAME = default_training_dataset_contract().target_name


class PODCoarseningFittable(PODCoarseningStrategy):
    """`PODCoarseningStrategy`, usable directly as a `FitJobConfig` job's `[model]`.

    Subclasses `PODCoarseningStrategy` (inheritance, not composition/wrapping)
    so the fitted checkpoint's buffer key stays exactly `_basis` — no wrapper
    attribute prefix in `state_dict()`. `register_buffer("_basis", ...)`
    happens in the parent's `fit()`, called on `self`, so a reconstructed
    instance round-trips through
    `composition/preconditioners/factory.py::_load_fitted_pod_coarsening`
    (`isinstance(model, PODCoarseningStrategy)`, `state_dict` shape) exactly
    like a bare `PODCoarseningStrategy` would — `isinstance` follows
    inheritance, and the checkpoint format is identical either way.

    Complexity: on top of the inherited SVD-fit cost (see
    `PODCoarseningConfig`'s complexity note — `O(min(m, n) * m * n)` for `m`
    snapshots, `n` DOFs), this class's `fit()` pays one extra O(dataset size)
    pass to materialize the dataloader's batches into a single snapshot
    tensor before delegating — the single-pass, never-`list()`-the-whole-
    dataloader concatenation described in `fit()`'s own docstring.
    """

    def __init__(self, rank: float, target_name: str = _DEFAULT_TARGET_NAME) -> None:
        """Store the target rank and dataloader batch key; unfitted until `fit()`.

        Args:
            rank: Fixed mode count (int) or minimum cumulative captured
                energy (float in (0, 1]) — forwarded unchanged to
                `PODCoarseningStrategy.__init__`.
            target_name: Batch key (under `batch["targets"]`) supplying the
                snapshot ensemble to fit from. Defaults to
                `RuntimeDatasetContract.target_name`
                (`composition/assignments/runtime_dataset_contract.py`) —
                the same target key every other assignment in this repo binds
                its supervised target array to.
        """
        super().__init__(rank=rank)
        self._target_name = target_name

    @property
    def hparams(self) -> dict[str, Any]:
        """Minimal Lightning-shaped `hparams` for dlkit's tracking param logger.

        `dlkit.engine.tracking.settings_logger.SettingsLogger.log_model_parameters`
        unconditionally reads `model.hparams` — populated, for the ordinary
        trainer-backed path, by `save_hyperparameters()` inside
        `CoreLightningWrapper`. `OneShotFitExecutor` runs this model directly
        with no Lightning wrapper at all (no `Trainer`, no wrapping step), so
        without this property the tracking decorator's param-logging step
        raises `AttributeError` before `fit()` ever runs — not
        POD-specific, but a gap in any `Fittable` model used through
        `FitJobConfig` today.
        """
        return {"rank": self._rank}

    def fit(
        self,
        snapshots: torch.Tensor | Iterable[Any],
        row_scales: torch.Tensor | None = None,
        inner_product: torch.Tensor | None = None,
    ) -> None:
        """Fit from a raw tensor, or by materializing a dataloader's batches first.

        An LSP-compatible override of `PODCoarseningStrategy.fit` — same
        parameters, same defaults, so this class stays substitutable
        wherever a plain `PODCoarseningStrategy` is expected (e.g.
        `factory.py`'s inline-fit call site). Anything other than a
        `torch.Tensor` is treated as a dlkit training dataloader — each batch's
        `["targets"][target_name]` entry is concatenated into one snapshot
        ensemble before delegating to the tensor path, satisfying
        `dlkit.engine.training.fittable.Fittable.fit(dataloader)`. Snapshot rows
        are fitted as stored: any row scaling happens at generation, with the
        matrix that generated them.

        Args:
            snapshots: Either the snapshot ensemble directly (shape
                `(n_samples, n_dofs)`), or a dataloader yielding batches
                whose `["targets"][target_name]` entry is this run's
                snapshot array (e.g. a `solutions-cgN` dataset's `y`).
            row_scales: Explicit per-snapshot row scale, forwarded straight
                to `PODCoarseningStrategy.fit` on both paths.
            inner_product: Dense SPD weight `W` of the POD inner product,
                forwarded straight to `PODCoarseningStrategy.fit` on both paths;
                `None` fits the Euclidean basis.

        Raises:
            ValueError: If `snapshots` is a dataloader that yields no batches.
        """
        if isinstance(snapshots, torch.Tensor):
            super().fit(snapshots, row_scales=row_scales, inner_product=inner_product)
            return
        # Single pass over the dataloader — never list(snapshots). Each batch is a
        # pin_memory'd TensorDict; materializing the whole dataloader up front would
        # hold every batch's pinned buffers alive simultaneously (the entire dataset,
        # doubled for the unused "inputs" field) instead of releasing each one as its
        # target tensor is extracted and copied off. `.detach().clone()` (not `.cpu()`,
        # which is a no-op view for an already-CPU tensor) forces a fresh, unpinned
        # allocation so the source batch's pinned storage can actually be freed.
        chunks = [
            torch.as_tensor(batch["targets"][self._target_name]).detach().clone()
            for batch in snapshots
        ]
        if not chunks:
            raise ValueError("PODCoarseningFittable.fit() received an empty dataloader.")
        # Moved to the resolved compute device (not left on CPU, where the dataloader
        # put it): the SVD (torch.linalg.svd, in compute_pod_basis) runs wherever
        # `concatenated` lives, and resolve_device() is the same GPU-if-available
        # convention comparison_run.py/domain/solver/comparison.py already use.
        concatenated = torch.cat(chunks, dim=0).to(resolve_device())
        super().fit(concatenated, row_scales=row_scales, inner_product=inner_product)
