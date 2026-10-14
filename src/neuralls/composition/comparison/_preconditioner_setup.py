"""Preconditioner creation, scheduling, input binding, and stopping criterion registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torchalg.preconditioners.base import BindableInputs, Preconditioner, StaticSetupPreconditioner

from neuralls.composition.comparison.models import PlannedSystem
from neuralls.composition.preconditioners.checkpoint_store import NO_PRELOADED, CheckpointStore
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.preconditioners.preloading import CheckpointLoaders, preload_checkpoints
from neuralls.composition.preconditioners.schedule import (
    PreconditionerScheduleConfig,
    create_scheduled_preconditioner,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
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
    """Cache of already-built-and-setup preconditioners for one matrix dataset.

    Reuse is correct by construction from `torchalg`'s own role interface:

    - A `StaticSetupPreconditioner`'s `setup()` never reads the matrix (e.g. a
      checkpoint-backed neural preconditioner), so one build serves every
      system of the dataset.
    - A `MatrixSetupPreconditioner`'s build depends on the matrix, so it is
      stored per `MatrixId` and only served to systems on that same matrix.

    A preconditioner name must mean one preconditioner within a dataset
    (`validate_dataset_specs_consistent`), which is what makes serving by name
    safe. Systems get a `SystemPreconditionerCache` view via `for_system`.
    """

    def __init__(self) -> None:
        self._static: dict[str, _CachedPreconditioner] = {}
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
        """Drop the matrix-dependent builds for ``matrix`` (reusable only on that matrix).

        Called after the matrix's last system executed, so a dataset with many
        matrices never keeps every matrix's AMG hierarchy or factorization
        resident. Static-setup entries are kept: they serve the whole dataset.
        """
        self._by_matrix.pop(matrix, None)

    def lookup(self, name: str, matrix: MatrixId) -> _CachedPreconditioner | None:
        cached = self._static.get(name)
        return cached if cached is not None else self._by_matrix.get(matrix, {}).get(name)

    def store(self, name: str, matrix: MatrixId, cached: _CachedPreconditioner) -> None:
        if isinstance(cached.preconditioner, StaticSetupPreconditioner):
            self._static[name] = cached
        else:
            self._by_matrix.setdefault(matrix, {})[name] = cached

    def shared_names(self) -> frozenset[str]:
        """Names whose setup this cache shares across every system it served.

        A static-setup name is shared dataset-wide. A matrix-setup name is
        shared across all systems only when they all solved on one matrix;
        with several matrices it is rebuilt per matrix, so its setup recurs
        and projecting it as a one-off would understate its cost. Used after
        execution to answer `SetupSharingPolicy.share_setup`.
        """
        if self._released_shared_names is not None:
            return self._released_shared_names
        shared = set(self._static)
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
        self._static.clear()
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

    def preload(self, matrix: torch.Tensor, config: PreconditionerConfig) -> None:
        """Load the checkpoints ``config`` depends on, outside any timed region.

        Loading a checkpoint is file I/O, not part of a preconditioner's
        algorithmic setup cost, and a checkpoint can serve many builds (every
        matrix of a dataset, every timing repeat); so it is loaded once, here,
        before the timed build, and the build borrows the loaded object. A
        no-op for configs without checkpoints and for a service built without a
        store.

        Args:
            matrix: System matrix; a fitted POD basis is moved to its dtype/device.
            config: Preconditioner configuration.
        """
        if self._checkpoints is None:
            return
        preload_checkpoints(
            config, matrix, self._checkpoints, CheckpointLoaders.default(self._adapter)
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
        return create_preconditioner(matrix, config, adapter=self._adapter, preloaded=preloaded)


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


def _load_extra_data(
    matrix_path: Path,
    name_to_position: dict[str, int],
    matrix_index: int,
) -> dict[str, torch.Tensor]:
    """Load named extra arrays from a dataset dir using declaration-order positions.

    Args:
        matrix_path: Dataset directory containing ``parameters_N`` array files.
        name_to_position: Mapping from semantic name to its positional index.
        matrix_index: Sample index to extract from each parameter array.

    Returns:
        Dict mapping names to loaded arrays (entries whose position exceeds the
        available parameter count are silently skipped).
    """
    arrays = load_training_arrays(matrix_path)
    return {
        name: torch.tensor(load_array_source_sample(arrays.parameter_sources[pos], matrix_index))
        for name, pos in name_to_position.items()
        if pos < len(arrays.parameter_sources)
    }


def _load_and_bind_extra_inputs(
    scheduled_preconditioners: dict[str, Preconditioner],
    matrix: torch.Tensor,
    matrix_path: Path,
    matrix_index: int,
) -> None:
    """Load extra parameter arrays and bind them alongside the system matrix.

    Args:
        scheduled_preconditioners: Preconditioners to inspect and bind.
        matrix: System matrix — always bound as ``"matrix"``.
        matrix_path: Dataset directory; loading is skipped when not a directory.
        matrix_index: Sample index used when extracting from parameter arrays.
    """
    name_to_position = _collect_extra_name_positions(scheduled_preconditioners)
    extra_data = (
        _load_extra_data(matrix_path, name_to_position, matrix_index)
        if name_to_position and matrix_path.is_dir()
        else {}
    )
    extra_data = {
        name: value.to(dtype=matrix.dtype, device=matrix.device)
        for name, value in extra_data.items()
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
