"""dlkit `Fittable` adapter over torchalg's `PODCoarseningStrategy`.

Lives in `composition/preconditioners/` (not `domain/solver/`) per
`domain/solver/README.md`'s boundary rule: all preconditioner algorithms live
in `torchalg`, and `neuralls.domain.solver` production code must not hold
preconditioner algorithm classes. This adapter holds no new math at all — it
inherits `build_transfer`/`compute_pod_basis` from `PODCoarseningStrategy`
unchanged — it exists solely so dlkit's job loader (`[model] module_path`,
resolved via `importlib`) has a concrete, importable class to construct for a
`run.type = "fit"` (POD-2G) assignment, mirroring
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

**Weighting is intentionally restricted to matrix-free schemes** (`raw`,
`power_norm` with `metric="l2"`). A `run.type = "fit"` job's entire premise
(`FitJobConfig`'s own docstring: "model and data are required... training
stays unset") is a one-shot, deterministic, checkpoint-reusable fit from a
*dataset alone* — the system matrix is a property of whichever comparison
later reconstructs and uses the checkpoint, not of the fit itself. Weighting
schemes that need the matrix (`power_norm` with `metric="a"`,
`smoother_persistence`) are only meaningful where the matrix is already a
natural, always-available input — `factory.py`'s inline
`PODCoarseningConfig` fit, used directly in a comparison, not threaded
through a reusable, matrix-agnostic fit-job artifact. See
`composition/preconditioners/_weighting.py::weighting_needs_matrix`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from loguru import logger
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.preconditioners._weighting import (
    resolve_row_scales,
    weighting_needs_matrix,
)
from neuralls.platform.config.models.preconditioner import (
    RawWeightingConfig,
    SnapshotWeightingConfig,
    parse_snapshot_weighting_config,
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
    """

    def __init__(
        self,
        rank: float,
        target_name: str = _DEFAULT_TARGET_NAME,
        weighting: SnapshotWeightingConfig | dict[str, Any] | None = None,
    ) -> None:
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
            weighting: Per-snapshot row scaling to apply before the SVD —
                see `SnapshotWeightingConfig`. Accepts a raw dict, since
                dlkit's `[model]` table (`extra="allow"`) passes unrecognized
                TOML keys through as plain kwargs, not validated model
                instances — coerced via `parse_snapshot_weighting_config`.
                `None` (default) is `RawWeightingConfig`, reproducing the
                original unweighted fit. Must not require the system matrix
                (see module docstring) — this is a fit *job*, not an inline
                comparison-time fit; use `PODCoarseningConfig.weighting`
                directly in a comparison for matrix-dependent schemes.

        Raises:
            ValueError: If `weighting` requires the system matrix.
        """
        super().__init__(rank=rank)
        self._target_name = target_name
        self._weighting = (
            parse_snapshot_weighting_config(weighting)
            if weighting is not None
            else RawWeightingConfig()
        )
        if weighting_needs_matrix(self._weighting):
            raise ValueError(
                f"{type(self._weighting).__name__} requires the system matrix, which a "
                "run.type='fit' job does not have — PODCoarseningFittable only supports "
                "matrix-free weighting (raw, power_norm with metric='l2'). Use "
                "PODCoarseningConfig.weighting directly in a comparison's preconditioner "
                "config instead, where the matrix is already available."
            )

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
        return {"rank": self._rank, "weighting": self._weighting.method}

    def fit(
        self, snapshots: torch.Tensor | Iterable[Any], row_scales: torch.Tensor | None = None
    ) -> None:
        """Fit from a raw tensor, or by materializing a dataloader's batches first.

        An LSP-compatible override of `PODCoarseningStrategy.fit` — same
        parameters, same defaults, so this class stays substitutable
        wherever a plain `PODCoarseningStrategy` is expected (e.g.
        `factory.py`'s inline-fit call site, tensor + explicit `row_scales`
        forwarded unchanged). Anything other than a `torch.Tensor` is
        treated as a dlkit training dataloader — each batch's
        `["targets"][target_name]` entry is concatenated into one snapshot
        ensemble, then (when `row_scales` wasn't already given explicitly)
        weighted per `self._weighting` — always matrix-free, per `__init__`'s
        validation — before delegating to the tensor path, satisfying
        `dlkit.engine.training.fittable.Fittable.fit(dataloader)`.

        Args:
            snapshots: Either the snapshot ensemble directly (shape
                `(n_samples, n_dofs)`), or a dataloader yielding batches
                whose `["targets"][target_name]` entry is this run's
                snapshot array (e.g. a `solutions-cgN` dataset's `y`).
            row_scales: Explicit per-snapshot row scale, forwarded straight
                to `PODCoarseningStrategy.fit`. When `snapshots` is a
                dataloader and this is left `None` (the normal case for a
                `run.type = "fit"` job), it is instead computed from
                `self._weighting`.

        Raises:
            ValueError: If `snapshots` is a dataloader that yields no batches.
        """
        logger.warning(
            "DIAGNOSTIC entered PODCoarseningFittable.fit(), tensor input={}",
            isinstance(snapshots, torch.Tensor),
        )
        try:
            if isinstance(snapshots, torch.Tensor):
                super().fit(snapshots, row_scales=row_scales)
                return
            logger.warning("DIAGNOSTIC materializing dataloader batches")
            batches = list(snapshots)
            if not batches:
                raise ValueError("PODCoarseningFittable.fit() received an empty dataloader.")
            logger.warning("DIAGNOSTIC materialized {} batches, concatenating", len(batches))
            concatenated = torch.cat(
                [torch.as_tensor(batch["targets"][self._target_name]) for batch in batches], dim=0
            )
            logger.warning(
                "DIAGNOSTIC pre-SVD snapshot tensor: shape={} dtype={} device={}",
                tuple(concatenated.shape),
                concatenated.dtype,
                concatenated.device,
            )
            if row_scales is None:
                row_scales = resolve_row_scales(self._weighting, concatenated, matrix=None)
            logger.warning("DIAGNOSTIC calling PODCoarseningStrategy.fit() (the SVD)")
            super().fit(concatenated, row_scales=row_scales)
        except Exception:
            logger.opt(exception=True).error("DIAGNOSTIC traceback for PODCoarseningFittable.fit()")
            raise
