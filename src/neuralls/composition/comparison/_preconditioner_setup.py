"""Preconditioner creation, scheduling, input binding, and stopping criterion registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torchalg.preconditioners.base import BindableInputs, Preconditioner

from neuralls.composition.comparison.models import PlannedSystem
from neuralls.composition.preconditioners.checkpoint_store import NO_PRELOADED, CheckpointStore
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.preconditioners.preloading import CheckpointLoaders, preload_checkpoints
from neuralls.composition.preconditioners.schedule import (
    PreconditionerScheduleConfig,
    create_scheduled_preconditioner,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
)
from neuralls.platform.storage.dataset_readers import (
    load_dense_training_arrays,
    load_parameter_arrays,
)
from neuralls.platform.storage.training_artifacts import (
    load_array_source_sample,
    load_training_arrays,
)
from neuralls.shared.types import MatrixId

if TYPE_CHECKING:
    from torchalg.preconditioners.ports import PredictorAdapter

# Arrays always available from LinearSystem — never need loading from disk.
_SYSTEM_ARRAYS_ALWAYS_AVAILABLE: frozenset[str] = frozenset({"matrix"})

StoppingCriterion = Literal["tolerance", "fixed_iterations"]

# Registry mapping config-level stopping criterion names to internal literals.
_STOPPING_CRITERION_REGISTRY: dict[str, StoppingCriterion] = {
    "tolerance": "tolerance",
    "residual_norm": "tolerance",
    "fixed": "fixed_iterations",
    "fixed_iterations": "fixed_iterations",
}


def register_stopping_criterion(name: str, criterion: StoppingCriterion) -> None:
    """Register a stopping criterion name mapping.

    Args:
        name: Config-level criterion name (case-insensitive).
        criterion: Internal typed literal — ``"tolerance"`` or ``"fixed_iterations"``.
    """
    _STOPPING_CRITERION_REGISTRY[name.lower()] = criterion


def _map_stopping_criterion(name: str) -> StoppingCriterion:
    """Map a string stopping criterion to the typed literal.

    Args:
        name: Stopping criterion name from config (e.g. ``"residual_norm"``).

    Returns:
        Typed literal ``"tolerance"`` or ``"fixed_iterations"``.

    Raises:
        ValueError: If the criterion name is not registered.
    """
    normalized = name.lower()
    criterion = _STOPPING_CRITERION_REGISTRY.get(normalized)
    if criterion is None:
        valid_names = ", ".join(sorted(_STOPPING_CRITERION_REGISTRY.keys()))
        raise ValueError(f"Unknown stopping criterion: '{name}'. Valid options: {valid_names}")
    return criterion


@dataclass(frozen=True)
class _CachedPreconditioner:
    """One preconditioner's already-built-and-ready object, plus its real one-time cost.

    Stored instead of re-measuring on a cache hit — copying these values
    forward (never reporting 0.0) is what lets a reused result still show
    its true setup cost.
    """

    preconditioner: Preconditioner
    setup_cost_seconds: float
    setup_peak_memory_bytes: int | None


def matrix_key(system: PlannedSystem) -> MatrixId:
    """The matrix a planned system solves on, for setup-cache scoping."""
    return system.matrix.matrix_id


class DatasetPreconditionerCache:
    """Legacy cache of prepared preconditioners, always scoped by matrix.

    Static and matrix-dependent methods both receive a distinct measured setup
    event on every matrix.  This cache therefore shares prepared state only
    among sibling RHS systems with the same `MatrixId`; checkpoint objects use
    the separate method-owned `CheckpointStore`.

    A preconditioner name must mean one preconditioner within a dataset
    (`validate_dataset_specs_consistent`), which is what makes serving by name
    safe. Systems get a `SystemPreconditionerCache` view via `for_system`.
    """

    def __init__(self) -> None:
        self._by_matrix: dict[MatrixId, dict[str, _CachedPreconditioner]] = {}
        self._matrices_seen: set[MatrixId] = set()
        self._released_shared_names: frozenset[str] | None = None

    def for_system(
        self, matrix: MatrixId, checkpoints: CheckpointStore | None = None
    ) -> SystemPreconditionerCache:
        """The view one system on ``matrix`` reads and writes through.

        Args:
            matrix: The system's matrix.
            checkpoints: The dataset's checkpoint store the view hands to the
                system's builds; ``None`` makes each comparison use a private one.
        """
        self._matrices_seen.add(matrix)
        return SystemPreconditionerCache(self, matrix, checkpoints)

    def evict_matrix(self, matrix: MatrixId) -> None:
        """Drop every prepared build for ``matrix``.

        Called after the matrix's last system executed, so a dataset with many
        matrices never keeps every hierarchy or factorization resident.
        """
        self._by_matrix.pop(matrix, None)

    def lookup(self, name: str, matrix: MatrixId) -> _CachedPreconditioner | None:
        return self._by_matrix.get(matrix, {}).get(name)

    def store(self, name: str, matrix: MatrixId, cached: _CachedPreconditioner) -> None:
        self._by_matrix.setdefault(matrix, {})[name] = cached

    def shared_names(self) -> frozenset[str]:
        """Names whose setup this cache shares across every system it served.

        A name is shared across every system only when the cache served one
        matrix.  With several matrices every method is rebuilt per matrix.
        """
        if self._released_shared_names is not None:
            return self._released_shared_names
        shared: set[str] = set()
        if len(self._matrices_seen) == 1:
            shared.update(name for names in self._by_matrix.values() for name in names)
        return frozenset(shared)

    def release(self) -> None:
        """Drop every built preconditioner (and the weights/hierarchies it holds).

        Called once the dataset's last system has executed, so a sweep over
        many datasets never keeps all their models and AMG hierarchies in memory
        at the same time. `shared_names()` is frozen first because the group
        report reads it after execution; any later lookup is a miss.
        """
        self._released_shared_names = self.shared_names()
        self._by_matrix.clear()


class SystemPreconditionerCache:
    """One system's view of its dataset's `DatasetPreconditionerCache`.

    Binds the system's matrix so `compare_preconditioners` and the per-
    preconditioner runner need only a name to read or write an entry.
    """

    def __init__(
        self,
        store: DatasetPreconditionerCache,
        matrix: MatrixId,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        self._store = store
        self._matrix = matrix
        self.checkpoints = checkpoints
        """The dataset's checkpoint store, owned by its `DatasetScope` (``None``: none shared)."""

    def get(self, name: str) -> _CachedPreconditioner | None:
        """Return the cached entry for preconditioner `name`, or `None` on a miss."""
        return self._store.lookup(name, self._matrix)

    def put(
        self,
        name: str,
        preconditioner: Preconditioner,
        *,
        setup_cost_seconds: float,
        setup_peak_memory_bytes: int | None,
    ) -> None:
        """Store `preconditioner` for reuse by later systems whose scope it is valid for."""
        self._store.store(
            name,
            self._matrix,
            _CachedPreconditioner(
                preconditioner=preconditioner,
                setup_cost_seconds=setup_cost_seconds,
                setup_peak_memory_bytes=setup_peak_memory_bytes,
            ),
        )


@dataclass(slots=True)
class DatasetScope:
    """Everything one `matrix_dataset` keeps alive across its systems, released together.

    The preconditioner cache serves built objects; the checkpoint store owns the
    loaded checkpoints they borrow. Keeping both lifetimes in one object means
    they cannot be released out of step.
    """

    cache: DatasetPreconditionerCache = field(default_factory=DatasetPreconditionerCache)
    checkpoints: CheckpointStore = field(default_factory=CheckpointStore)

    def for_system(self, matrix: MatrixId) -> SystemPreconditionerCache:
        """The view one system on ``matrix`` reads and writes through."""
        return self.cache.for_system(matrix, self.checkpoints)

    def release(self) -> None:
        """Drop the cached builds, then the checkpoints they borrowed."""
        self.cache.release()
        self.checkpoints.release()


class PreconditionerService:
    """Service for creating preconditioners via the composition factory.

    Args:
        adapter: Optional adapter for neural preconditioners (dependency injection for testing).
        checkpoints: Where checkpoints are loaded ahead of a build (see `preload`).
            ``None`` means nothing is preloaded: every checkpoint is then read
            inside the build, and so inside its timed region.
    """

    def __init__(
        self, adapter: PredictorAdapter | None = None, checkpoints: CheckpointStore | None = None
    ) -> None:
        self._adapter = adapter
        self._checkpoints = checkpoints
        self._staged_fit_data: dict[str, tuple[np.ndarray, ...]] = {}

    def preload(self, matrix: torch.Tensor, config: PreconditionerConfig) -> None:
        """Load the checkpoints ``config`` depends on, outside any timed region.

        Loading a checkpoint is file I/O, not part of a preconditioner's
        algorithmic setup cost, and a checkpoint can serve many builds (every
        matrix of this method workload, every timing repeat); so it is loaded once, here,
        before the timed build, and the build borrows the loaded object. A
        no-op for configs without checkpoints and for a service built without a
        store.

        Args:
            matrix: System matrix; a fitted POD basis is moved to its dtype/device.
            config: Preconditioner configuration.
        """
        if self._checkpoints is not None:
            preload_checkpoints(
                config, matrix, self._checkpoints, CheckpointLoaders.default(self._adapter)
            )
        if isinstance(config, AMGPreconditionerConfig):
            coarsening = config.coarsening
            if (
                isinstance(coarsening, PODCoarseningConfig)
                and coarsening.active_checkpoint_path is None
            ):
                if config.name not in self._staged_fit_data:
                    self._staged_fit_data[config.name] = tuple(
                        load_dense_training_arrays(coarsening.dataset_dir)
                    )
            elif (
                isinstance(coarsening, NeuralPODCoarseningConfig)
                and config.name not in self._staged_fit_data
            ):
                self._staged_fit_data[config.name] = tuple(
                    load_parameter_arrays(coarsening.dataset_dir)
                )

    def create_preconditioner(
        self,
        matrix: torch.Tensor,
        config: PreconditionerConfig,
    ) -> Preconditioner:
        """Create a single preconditioner, borrowing whatever `preload` already loaded.

        Args:
            matrix: System matrix to precondition (shape: n x n).
            config: Preconditioner configuration.

        Returns:
            Preconditioner instance.
        """
        preloaded = self._checkpoints if self._checkpoints is not None else NO_PRELOADED
        return create_preconditioner(
            matrix,
            config,
            adapter=self._adapter,
            preloaded=preloaded,
            staged_fit_data=self._staged_fit_data.get(config.name),
        )


def _create_scheduled_preconditioners(
    preconditioner_configs: Sequence[PreconditionerConfig],
    matrix: torch.Tensor,
    base_preconditioners: dict[str, Any],
) -> dict[str, Preconditioner]:
    """Wrap base preconditioners with scheduling if configured.

    Args:
        preconditioner_configs: Config objects carrying scheduling parameters.
        matrix: System matrix (needed for fallback creation).
        base_preconditioners: Already-created preconditioner instances.

    Returns:
        Dict mapping names to scheduled (or unscheduled) preconditioner instances.
    """
    scheduled: dict[str, Preconditioner] = {}
    for cfg in preconditioner_configs:
        primary = base_preconditioners[cfg.name]
        schedule = PreconditionerScheduleConfig(
            start_iter=cfg.start_iter,
            limit_iters=cfg.limit_iters,
            fallback=cfg.fallback,
        )
        scheduled[cfg.name] = create_scheduled_preconditioner(
            primary=primary,
            schedule=schedule,
            matrix=matrix,
        )
    return scheduled


def _collect_extra_name_positions(
    preconditioners: dict[str, Preconditioner],
) -> dict[str, int]:
    """Map extra-input name → declaration-order index across all bindable preconditioners.

    Args:
        preconditioners: Scheduled preconditioner map to inspect.

    Returns:
        Dict mapping each extra name to its position in ``extra_input_names``
        (first occurrence wins, preserving training-config declaration order).
    """
    positions: dict[str, int] = {}
    for p in preconditioners.values():
        if not isinstance(p, BindableInputs):
            continue
        for i, name in enumerate(p.extra_input_names):
            if name not in _SYSTEM_ARRAYS_ALWAYS_AVAILABLE:
                positions.setdefault(name, i)
    return positions


def _load_extra_inputs(matrix_path: Path, matrix_index: int) -> tuple[np.ndarray, ...]:
    """Read every parameter array for one matrix outside online timing.

    Args:
        matrix_path: Dataset directory containing ``parameters_N`` array files.
        matrix_index: Sample index to extract from each parameter array.

    Returns:
        Parameter arrays in manifest order. Torch tensor construction,
        device/dtype conversion and semantic binding deliberately happen
        later, inside setup timing.
    """
    if not matrix_path.is_dir():
        return ()
    try:
        arrays = load_training_arrays(matrix_path)
    except FileNotFoundError:
        return ()
    return tuple(
        np.array(load_array_source_sample(source, matrix_index), copy=True)
        for source in arrays.parameter_sources
    )


def _bind_extra_inputs(
    scheduled_preconditioners: dict[str, Preconditioner],
    matrix: torch.Tensor,
    staged_inputs: tuple[np.ndarray, ...],
) -> None:
    """Convert staged parameter arrays and bind them inside setup timing.

    Args:
        scheduled_preconditioners: Preconditioners to inspect and bind.
        matrix: System matrix — always bound as ``"matrix"``.
        staged_inputs: Parameter arrays read before setup, in manifest order.
    """
    name_to_position = _collect_extra_name_positions(scheduled_preconditioners)
    extra_data = {
        name: torch.as_tensor(staged_inputs[position], dtype=matrix.dtype, device=matrix.device)
        for name, position in name_to_position.items()
        if position < len(staged_inputs)
    }
    _bind_system_inputs(scheduled_preconditioners, {"matrix": matrix, **extra_data})


def _bind_system_inputs(
    preconditioners: dict[str, Preconditioner],
    system_data: dict[str, torch.Tensor],
) -> None:
    """Bind dataset-sourced extra inputs to preconditioners that declare them.

    Reads extra_input_names from each preconditioner. For each name present in
    system_data, calls bind_inputs() so the preconditioner can forward them to
    the model without changing the CG-facing apply(residual) interface.

    Args:
        preconditioners: Map of name to preconditioner.
        system_data: Available named arrays (always includes ``"matrix"``).
    """
    for precond in preconditioners.values():
        if not isinstance(precond, BindableInputs):
            continue
        needed = {k: v for k, v in system_data.items() if k in precond.extra_input_names}
        if needed:
            precond.bind_inputs(**needed)
