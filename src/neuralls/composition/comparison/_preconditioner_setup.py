"""Preconditioner creation, scheduling, input binding, and stopping criterion registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torchalg.preconditioners.base import BindableInputs, Preconditioner, StaticSetupPreconditioner

from neuralls.composition.comparison.models import GroupMatrixIdentity, MultiMatrixDataset
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.preconditioners.schedule import (
    PreconditionerScheduleConfig,
    create_scheduled_preconditioner,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.storage.training_artifacts import (
    load_array_source_sample,
    load_training_arrays,
)

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


def _allows_matrix_dependent_sharing(identity: GroupMatrixIdentity) -> bool:
    """Whether a `MatrixSetupPreconditioner` can share its setup across this group.

    True for a `SingleMatrixDataset` (nothing else it could be) or a
    `MultiMatrixDataset` whose members all resolved to the same matrix id.
    """
    return not isinstance(identity, MultiMatrixDataset) or len(identity.resolved_indices) == 1


class GroupPreconditionerCache:
    """Per-comparison-group cache of already-built-and-setup preconditioners.

    Scoped to one `ComparisonGroup` (comparisons sharing a matrix/RHS-recipe
    family, whether hand-written or `comparison_sweeps`-expanded) —
    constructed once per group by the caller and passed into every
    member's `compare_preconditioners` call, so a preconditioner built for
    the first member can be reused by the rest whenever doing so is
    provably valid:

    - Always valid for a `torchalg.StaticSetupPreconditioner` (its `setup()`
      never reads the matrix at all — see that class's docstring).
    - Valid for a `torchalg.MatrixSetupPreconditioner` only when every
      member in this group resolves to the *same* matrix.

    `put` silently does nothing when neither condition holds, so a classical
    preconditioner in a matrix-varying group is simply never cached — it
    keeps rebuilding fresh every member, exactly like today, with no risk
    of ever serving a stale build.
    """

    def __init__(self, *, matrix_identity: GroupMatrixIdentity) -> None:
        """Create an empty cache scoped to one comparison group.

        Args:
            matrix_identity: This group's `GroupMatrixIdentity`, as computed
                by group discovery — never inferred locally from a flag.
        """
        self._matrix_identity = matrix_identity
        self._entries: dict[str, _CachedPreconditioner] = {}

    def get(self, name: str) -> _CachedPreconditioner | None:
        """Return the cached entry for preconditioner `name`, or `None` on a miss."""
        return self._entries.get(name)

    def put(
        self,
        name: str,
        preconditioner: Preconditioner,
        *,
        setup_cost_seconds: float,
        setup_peak_memory_bytes: int | None,
    ) -> None:
        """Store `preconditioner` for reuse by later siblings, if sharing is valid for it.

        A no-op when neither sharing condition holds for `preconditioner` —
        the next sibling will see a cache miss and rebuild fresh, same as if
        no cache existed at all.
        """
        if not (
            isinstance(preconditioner, StaticSetupPreconditioner)
            or _allows_matrix_dependent_sharing(self._matrix_identity)
        ):
            return
        self._entries[name] = _CachedPreconditioner(
            preconditioner=preconditioner,
            setup_cost_seconds=setup_cost_seconds,
            setup_peak_memory_bytes=setup_peak_memory_bytes,
        )

    def shared_names(self) -> frozenset[str]:
        """Names of every preconditioner this cache actually shared across siblings.

        Used after a sweep group finishes executing to answer
        `SetupSharingPolicy.share_setup` per preconditioner — by then the
        live objects are out of scope, but the cache's own record of what it
        successfully stored survives as the fact of record.
        """
        return frozenset(self._entries)


class PreconditionerService:
    """Service for creating and managing preconditioners via the composition factory.

    Args:
        adapter: Optional adapter for neural preconditioners (dependency injection for testing).
    """

    def __init__(self, adapter: PredictorAdapter | None = None) -> None:
        self._adapter = adapter

    def create_preconditioner(
        self,
        matrix: torch.Tensor,
        config: PreconditionerConfig,
    ) -> Preconditioner:
        """Create a single preconditioner.

        Args:
            matrix: System matrix to precondition (shape: n x n).
            config: Preconditioner configuration.

        Returns:
            Preconditioner instance.
        """
        return create_preconditioner(matrix, config, adapter=self._adapter)


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
